from typing import Literal, Optional

import torch
import yaml
import lightning as L
from pydantic import BaseModel


from ..loss import FaceFormerLoss, VocaLoss
from .voca import Voca
from .audio2face import Audio2Mesh
from .song2face import Song2Face
from .faceformer import Faceformer
from .extractor import MFCCExtractor, Wav2VecExtractor

from ..utils.renderer import Renderer, FaceMesh, images_to_video, save_audio


class ExpConfig(BaseModel):
    # dataset
    batch_size: int
    # model
    modelname: str
    one_hot_size: int
    feature_extractor: str
    sample_rate: int
    vertex_count: int
    split_frame: bool
    n_feature: int
    out_dim: int
    win_length: int
    hop_length: Optional[int] = None
    # training
    percision: str
    lr: float
    # loss
    loss: Optional[str] = None
    is_transformer: bool = False

    def fit_transformer(self) -> "ExpConfig":
        if self.modelname == "faceformer":
            self.split_frame = False
            self.batch_size = 1
            self.feature_extractor = None
            self.is_transformer = True

        return self

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)

    def exp_name(self):
        return f"{self.modelname}_{self.feature_extractor}_{self.lr}_{self.loss}_{self.percision}"


def get_model(modelname: Literal["voca", "audio2mesh", "song2face", "faceformer", "af_model"]):
    model_map = {
        "voca": Voca,
        "audio2mesh": Audio2Mesh,
        "song2face": Song2Face,
        "faceformer": Faceformer,
        # "af_model": AFModel,
    }
    return model_map[modelname]


def get_extractor(extractor: Literal["mfcc", "wav2vec"]):
    extractor_map = {
        "mfcc": MFCCExtractor,
        "wav2vec": Wav2VecExtractor,
        None: lambda *args, **kwargs: None,  # noqa
    }
    return extractor_map[extractor]


def get_loss_fn(modelname: str):
    if modelname == "faceformer":
        return FaceFormerLoss()
    return VocaLoss()


class Audio2FaceModel(L.LightningModule):
    def __init__(self, config: ExpConfig):
        super().__init__()
        self.model_name = config.modelname
        model = get_model(config.modelname)
        # fe
        self.fe_name = config.feature_extractor
        fe = get_extractor(config.feature_extractor)
        self.feature_extractor = fe(
            sample_rate=config.sample_rate,
            n_feature=config.n_feature,
            out_dim=config.out_dim,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=1024,
        )

        # model
        self.model = model(n_verts=config.vertex_count, n_onehot=config.one_hot_size)
        self.loss = get_loss_fn(config.modelname) if config.loss is None else config.loss
        self.lr = config.lr
        self.lr_weight_decay = self.lr / 10

        self.training_error = []
        self.validation_error = []
        self.validation_pred_verts = []
        self.validation_gt_verts = []

        self.predict_error = []
        self.predicted_verts = []

        self.save_hyperparameters()

    def forward(self, x, one_hot, template, **kwargs):
        if self.feature_extractor is None:
            return self.model(x, one_hot, template, **kwargs)

        feature = self.feature_extractor(x).detach()
        pred_verts = self.model(feature, one_hot, template, **kwargs)
        return pred_verts

    @torch.no_grad()
    def mse_error(self, pred, gt):
        pred = pred.view(-1, 5023 * 3)
        gt = gt.view(-1, 5023 * 3)
        with torch.no_grad():
            a = torch.mean((pred - gt) ** 2, axis=1)
            return torch.mean(a)

    def on_train_epoch_end(self):
        epoch_err = sum(self.training_error) / len(self.training_error)
        self.log("train/err", epoch_err, on_epoch=True, on_step=False)
        self.logger.experiment.add_scalar("train_epock/err", epoch_err, self.current_epoch)
        print(f"Epoch {self.current_epoch} train err: {epoch_err}")
        self.training_error.clear()

    def on_validation_epoch_end(self):
        epoch_err = sum(self.validation_error) / len(self.validation_error)
        self.log("val/err", epoch_err, on_epoch=True, on_step=False)
        self.logger.experiment.add_scalar("val_epock/err", epoch_err, self.current_epoch)
        print(f"Epoch {self.current_epoch} val error: {epoch_err}")
        self.validation_error.clear()

    def unpack_batch(self, batch):
        batch["verts"] = batch["verts"] * 100
        batch["template_vert"] = batch["template_vert"] * 100
        return batch["audio"], batch["one_hot"], batch["verts"], batch["template_vert"]

    def training_step(self, batch, batch_idx):
        # empty the cache
        torch.cuda.empty_cache()
        x, one_hot, gt, template = self.unpack_batch(batch)
        pred = self(x, one_hot, template)
        loss = self.loss(pred, gt)
        if self.global_step % 10 == 0:
            self.log_dict(loss, on_epoch=False, on_step=True, prog_bar=True)

        err = self.mse_error(pred, gt)
        self.training_error.append(err.item())
        return loss

    def validation_step(self, batch, batch_idx):
        x, one_hot, gt, template = self.unpack_batch(batch)
        pred = self(x, one_hot, template)
        loss = self.loss(pred, gt)
        self.log_dict(loss, on_epoch=False, on_step=True, prog_bar=True)

        err = self.mse_error(pred, gt)
        self.validation_error.append(err.item())
        # self.validation_pred_verts.append(pred / 100)
        # self.validation_gt_verts.append(gt / 100)

    def on_validation_end(self):
        return
        if self.model_name == "faceformer":
            self.validation_pred_verts = [item.squeeze(0) for item in self.validation_pred_verts]
            self.validation_gt_verts = [item.squeeze(0) for item in self.validation_gt_verts]

        predicted_verts = torch.cat(self.validation_pred_verts, axis=0)
        gt_verts = torch.cat(self.validation_gt_verts, axis=0)
        self.validation_pred_verts.clear()
        self.validation_gt_verts.clear()

        # select one sample to log
        random_idx = torch.randint(0, predicted_verts.shape[0], (1,))
        try:
            renderer = self.get_renderer()
            rendered_image = renderer.render(predicted_verts[random_idx].cpu().numpy())[0]
            gt_image = renderer.render(gt_verts[random_idx].cpu().numpy())[0]
        except Exception as e:
            print("Failed to render image", e)
            return

        if self.logger:
            self.logger.experiment.add_image("val/prediction", rendered_image.transpose(2, 0, 1), self.current_epoch)
            self.logger.experiment.add_image("val/gt", gt_image.transpose(2, 0, 1), self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.lr_weight_decay)
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, one_hot, gt, template = self.unpack_batch(batch)
        pred = self(x, one_hot, template)
        self.predict_error.append(self.mse_error(pred, gt).item())
        for item in pred:
            if self.model_name == "faceformer":
                self.predicted_verts.append(item / 100)
            else:
                self.predicted_verts.append(item.unsqueeze(0) / 100)
        self.predicted_audio = x
        return pred

    def get_renderer(self):
        if not hasattr(self, "renderer"):
            self.renderer = Renderer(FaceMesh.load("assets/FLAME_sample.obj"))
        return self.renderer

    def on_predict_epoch_end(self):
        # err
        epoch_rec_loss = sum(self.predict_error) / len(self.predict_error)
        print(f"Epoch {self.current_epoch} predict_rec_loss: {epoch_rec_loss}")
        self.predict_error.clear()

        # render video
        renderer = self.get_renderer()
        predicted_verts = torch.cat(self.predicted_verts, axis=0)
        self.predicted_verts.clear()

        rendered_image = renderer.render(predicted_verts.cpu().numpy())

        save_audio(self.predicted_audio, self.logger.log_dir)
        images_to_video(rendered_image, self.logger.log_dir)

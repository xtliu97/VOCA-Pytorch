import logging

import torch
import torchaudio
from torch import nn
from torchaudio.functional import resample
from transformers import Wav2Vec2Model, Wav2Vec2Processor


class MFCCExtractor(nn.Module):
    """
    Input shape: (batch, time)
    Output shape: (batch, out_dim, n_mfcc)
    """

    def __init__(
        self,
        sample_rate: int,
        n_feature: int,
        out_dim: int,
        win_length: int,
        hop_length: int = None,
        n_fft: int = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mfcc = n_feature
        self.out_dim = out_dim
        self.win_length = win_length
        self.hop_length = (
            hop_length if hop_length else win_length // 2
        )  # default hop_length is half of win_length
        self.n_fft = n_fft if n_fft else win_length  # default n_fft is win_length
        self.T = self._get_extractor()
        self.__running_for_first_time = True

    def _get_extractor(self):
        return torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "win_length": self.win_length,
            },
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.T(x).transpose(1, 2)
        if self.out_dim != x.shape[1]:
            if self.__running_for_first_time:
                logging.warning(
                    f"MFCCExtractor: Got tensor shape {x.shape}, resizing to {self.out_dim} using bilinear interpolation"
                )
                self.__running_for_first_time = False
            x = torch.nn.functional.interpolate(
                x.unsqueeze(1), size=(self.out_dim, self.n_mfcc), mode="bilinear"
            ).squeeze(1)
        return x


class Wav2VecExtractor(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_feature: int,
        out_dim: int,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.ori_sample_rate = sample_rate
        self.sample_rate = 16000
        self.out_dim = out_dim
        self.n_feature = n_feature

        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.model.freeze_feature_encoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = resample(x, self.ori_sample_rate, self.sample_rate)
        device = self.model.device
        x = self.processor(
            x, return_tensors="pt", padding=True, sampling_rate=self.sample_rate
        )
        x = self.model(x.input_values[0].to(device)).last_hidden_state
        x = x.transpose(1, 2)
        if self.out_dim != x.shape[1]:
            x = torch.nn.functional.interpolate(
                x.unsqueeze(1), size=(self.out_dim, self.n_feature), mode="bilinear"
            ).squeeze(1)
        return x

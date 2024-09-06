import pickle
from pathlib import Path
from typing import TypedDict, Literal, NamedTuple
from typing_extensions import Unpack
from functools import lru_cache

import torch
import torchaudio
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning as L
from rich import print

from .vocaset_split import training_subject, validation_subject, test_subject, training_sentence, validation_sentence


def load_pickle(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data


def load_npy_mmapped(npy_path: Path):
    return np.load(npy_path, mmap_mode="r")


VOCASET_AUDIO_TYPE = dict[str, dict[str, np.ndarray]]
VOCASET_VERTS_TYPE = np.ndarray
VOCASET_WAV_SEQ_TO_IDX_TYPE = dict[str, dict[str, dict[int, int]]]

VALID_PHASE = Literal["train", "val", "test", "all"]


def get_human_id_one_hot(human_id: str):
    all_human_id = [*training_subject, *validation_subject, *test_subject]
    one_hot = np.zeros(len(all_human_id))
    one_hot[all_human_id.index(human_id)] = 1
    return one_hot


@lru_cache(maxsize=20)
def get_template_vert(datapath: str | Path, human_id: str):
    template_verts_path = Path(datapath) / "templates.pkl"
    template_verts = load_pickle(template_verts_path)
    return template_verts[human_id]


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.int16:
        audio = audio / 32768
        return audio.astype(np.float32)
    else:
        raise f"Got audio with dtype {audio.dtype} when normalizing, expected np.int16"


class VocaItem(TypedDict):
    audio: torch.Tensor
    verts: torch.Tensor
    template_vert: torch.Tensor
    one_hot: torch.Tensor
    feature: torch.Tensor


class VocaClip(NamedTuple):
    human_id: str
    sentence_id: str
    clip_index: int
    data_verts_index: int


class DataSplitRecorder:
    def __init__(self, writable: bool = True) -> None:
        self.writable = writable
        self.train_list: list[VocaClip] = []
        self.val_list: list[VocaClip] = []
        self.test_list: list[VocaClip] = []

    def check_write_permission(func):
        def wrapper(self, *args, **kwargs):
            if self.writable:
                return func(self, *args, **kwargs)
            else:
                raise ValueError("This instance is read only")

        return wrapper

    @check_write_permission
    def add(self, voca_clip: VocaClip):
        if voca_clip.human_id in training_subject and voca_clip.sentence_id in training_sentence:
            self.train_list.append(voca_clip)
        elif voca_clip.human_id in validation_subject and voca_clip.sentence_id in validation_sentence:
            self.val_list.append(voca_clip)
        else:
            self.test_list.append(voca_clip)

    @check_write_permission
    def save(self, path: Path):
        def save_list(voca_clips: list[VocaClip], csv_path: Path):
            df = pd.DataFrame(
                voca_clips,
                columns=["human_id", "sentence_id", "clip_index", "data_verts_index"],
            )
            df.to_csv(csv_path, index=False)

        save_list(self.train_list, path / "train_list.csv")
        save_list(self.val_list, path / "val_list.csv")
        save_list(self.test_list, path / "test_list.csv")

    @staticmethod
    def generate_split_file(
        raw_audio: VOCASET_AUDIO_TYPE,
        subj_seq_to_idx: VOCASET_WAV_SEQ_TO_IDX_TYPE,
        *,
        save_path: Path | str,
    ):
        # create the split directory
        save_path = Path(save_path) / "split"
        save_path.mkdir(parents=True, exist_ok=True)

        data_split_recorder = DataSplitRecorder()

        # add clips to the recorder
        for clip_name, clip_data in raw_audio.items():
            if clip_name not in subj_seq_to_idx:
                continue
            for sentence_id in clip_data.keys():
                if sentence_id not in subj_seq_to_idx[clip_name]:
                    continue
                audio_idxs_mapping = subj_seq_to_idx[clip_name][sentence_id]
                for clip_index, seq_num in audio_idxs_mapping.items():
                    data_split_recorder.add(VocaClip(clip_name, sentence_id, clip_index, seq_num))
        data_split_recorder.save(save_path)

    @staticmethod
    def exists(path: Path):
        path = path / "split"
        return (
            (path / "train_list.csv").exists()
            and (path / "val_list.csv").exists()
            and (path / "test_list.csv").exists()
        )

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path) / "split"

        def _load_csv_to_list(name: str):
            df = pd.read_csv(name, header=0)
            return df.to_numpy().tolist()

        train_list = _load_csv_to_list(path / "train_list.csv")
        val_list = _load_csv_to_list(path / "val_list.csv")
        test_list = _load_csv_to_list(path / "test_list.csv")
        recorder = cls(write=False)
        recorder.train_list = train_list
        recorder.val_list = val_list
        recorder.test_list = test_list
        return recorder

    def get_datalist(self, phase: VALID_PHASE = "all"):
        match phase:
            case "train":
                return self.train_list
            case "val":
                return self.val_list
            case "test":
                return self.test_list
            case "all":
                return self.train_list + self.val_list + self.test_list


class ClipVocaSet(Dataset):
    def __init__(
        self,
        datapath: str | Path,
        phase: VALID_PHASE = "all",
        random_shift: bool = False,
        split_frame: bool = True,
        normalize_audio: bool = True,
    ):
        if not split_frame:
            assert random_shift is False, "random_shift is not supported when split_frame is False"

        self.phase = phase
        self.random_shift = random_shift
        self.datapath = Path(datapath).absolute()
        self.split_frame = split_frame
        self.normalize_audio = normalize_audio

        self.tempalte_verts: dict[str, np.ndarray] = load_pickle(self.datapath / "templates.pkl")
        self.raw_audio: VOCASET_AUDIO_TYPE = load_pickle(self.datapath / "raw_audio_fixed.pkl")
        self.data_verts: VOCASET_VERTS_TYPE = load_npy_mmapped(self.datapath / "data_verts.npy")
        self.wav_seq_to_idx: VOCASET_WAV_SEQ_TO_IDX_TYPE = load_pickle(self.datapath / "subj_seq_to_idx.pkl")

        if not DataSplitRecorder.exists(self.datapath):
            print("Generating split file...")
            DataSplitRecorder.generate_split_file(
                raw_audio=self.raw_audio,
                subj_seq_to_idx=self.wav_seq_to_idx,
                save_path=self.datapath,
            )

        self.split_recorder = DataSplitRecorder.load(self.datapath)
        self.datalist_raw = self.split_recorder.get_datalist(phase)
        if self.split_frame:
            self.datalist = self.datalist_raw
        else:
            unique_datalist = set()
            for data_info in self.datalist_raw:
                human_id, sentence_id, audio_index, data_verts_index = data_info
                unique_datalist.add((human_id, sentence_id))
            self.datalist = list(unique_datalist)

        print(f"Loaded {len(self.datalist)} frame data of phase {self.phase}")

    def __repr__(self) -> str:
        return f"""
ClipVocaSet(
    phase={self.phase},
    datapath={self.datapath},
    len={len(self.datalist)}
)"""

    def get_single_item(self, key: VocaClip) -> VocaItem:
        human_id, sentence_id, audio_index, data_verts_index = key

        audio = self.raw_audio[human_id][sentence_id]["audio"]
        sr = self.raw_audio[human_id][sentence_id]["sample_rate"]
        verts = self.data_verts[data_verts_index]
        if self.random_shift and self.phase == "train":
            random_shift = random.randint(-500, 500)  # 1%
        else:
            random_shift = 0
        audio_clip = get_audio_fragment(audio, audio_index, fps=60, sample_rate=sr, length=0.52, shift=random_shift)
        if self.normalize_audio:
            audio_clip = normalize_audio(audio_clip)

        return VocaItem(
            audio=audio_clip,
            verts=verts.astype(np.float32),
            template_vert=self.tempalte_verts[human_id].astype(np.float32),
            one_hot=get_human_id_one_hot(human_id).astype(np.float32),
        )

    def get_whole_clip(self, key) -> VocaItem:
        human_id, sentence_id = key
        audio_clip = self.raw_audio[human_id][sentence_id]["audio"]
        audio_idxs_mapping = self.wav_seq_to_idx[human_id][sentence_id]
        verts = [self.data_verts[seq_num] for seq_num in audio_idxs_mapping.values()]
        # resample to 16000
        if self.normalize_audio:
            audio_clip = normalize_audio(audio_clip)
        audio_clip = (
            torchaudio.functional.resample(torch.from_numpy(audio_clip), 22000, 16000).numpy().astype(np.float32)
        )
        return VocaItem(
            audio=audio_clip,
            verts=np.stack(verts).astype(np.float32),
            template_vert=self.tempalte_verts[human_id].astype(np.float32),
            one_hot=get_human_id_one_hot(human_id).astype(np.float32),
        )

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        if self.split_frame:
            return self.get_single_item(self.datalist[idx])
        else:
            return self.get_whole_clip(self.datalist[idx])

    def get_framedatas(self, target_human_id: str, target_sentence_id: str):
        res = []
        if self.split_frame:
            for data_info in self.datalist:
                if data_info[0] == target_human_id and data_info[1] == target_sentence_id:
                    res.append((self.get_single_item(data_info), data_info[2]))
            sorted(res, key=lambda x: x[1])
        else:
            for data_info in self.datalist:
                if data_info[0] == target_human_id and data_info[1] == target_sentence_id:
                    return [self.get_whole_clip(data_info)]

        return [x[0] for x in res]


class VocaDataModule(L.LightningDataModule):
    def __init__(
        self,
        datapath: str,
        batch_size: int = 32,
        num_workers: int = 4,
        random_shift: bool = False,
        split_frame: bool = True,
    ):
        super().__init__()
        self.datapath = datapath
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_shift = random_shift
        self.split_frame = split_frame

    def setup(self, stage: str = None):
        self.train_dataset = ClipVocaSet(
            self.datapath,
            phase="train",
            random_shift=self.random_shift,
            split_frame=self.split_frame,
        )
        self.val_dataset = ClipVocaSet(
            self.datapath,
            phase="val",
            random_shift=self.random_shift,
            split_frame=self.split_frame,
        )
        self.test_dataset = ClipVocaSet(
            self.datapath,
            phase="test",
            random_shift=self.random_shift,
            split_frame=self.split_frame,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def predict_dataloader(self, human_id: str, sentence_id: str):
        return DataLoader(
            self.test_dataset.get_framedatas(human_id, sentence_id),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class AduioParams(TypedDict):
    fps: int
    sample_rate: int
    length: float
    shift: int


def get_audio_fragment(audio: np.ndarray, idx: int, **audio_params: Unpack[AduioParams]) -> np.ndarray:
    _audio_dtype = audio.dtype  # keep the original dtype
    l_pad_samples = int(audio_params["sample_rate"] * audio_params["length"] / 2) + audio_params["shift"]
    n_pad_samples = int(audio_params["sample_rate"] * audio_params["length"] / 2)
    pad_audio = np.concatenate(
        [
            np.zeros(l_pad_samples, dtype=_audio_dtype),
            audio,
            np.zeros(2 * n_pad_samples, dtype=_audio_dtype),
        ]
    )
    start = idx * audio_params["sample_rate"] // audio_params["fps"]
    end = start + n_pad_samples * 2
    # check if the audio is long enough
    if end > len(pad_audio):
        print(f"Audio is not long enough to get fragment: {end} > {len(pad_audio)}")
        return None
    return pad_audio[start:end]

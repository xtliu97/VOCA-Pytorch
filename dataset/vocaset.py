import os
import pickle
from typing import TypedDict, Mapping, Literal
from typing_extensions import Unpack
from functools import lru_cache

import torch
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from rich import print


def load_pickle(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data


def load_npy_mmapped(npy_path):
    return np.load(npy_path, mmap_mode="r")


VOCASET_AUDIO_TYPE = Mapping[str, Mapping[str, np.ndarray]]
VOCASET_VERTS_TYPE = np.ndarray
VOCASET_WAV_SEQ_TO_IDX_TYPE = Mapping[str, Mapping[str, Mapping[int, int]]]

training_subject = [
    "FaceTalk_170728_03272_TA",
    "FaceTalk_170904_00128_TA",
    "FaceTalk_170725_00137_TA",
    "FaceTalk_170915_00223_TA",
    "FaceTalk_170811_03274_TA",
    "FaceTalk_170913_03279_TA",
    "FaceTalk_170904_03276_TA",
    "FaceTalk_170912_03278_TA",
]
training_sentence = [f"sentence{i:02d}" for i in range(1, 41)]
validation_subject = [
    "FaceTalk_170811_03275_TA",
    "FaceTalk_170908_03277_TA",
]
validation_sentence = [f"sentence{i:02d}" for i in range(21, 41)]
test_subject = ["FaceTalk_170809_00138_TA", "FaceTalk_170731_00024_TA"]


def get_human_id_one_hot(human_id: str):
    all_human_id = [*training_subject, *validation_subject, *test_subject]
    one_hot = np.zeros(len(all_human_id))
    one_hot[all_human_id.index(human_id)] = 1
    return one_hot


@lru_cache(maxsize=20)
def get_template_vert(datapath, human_id):
    template_verts_path = os.path.join(datapath, "templates.pkl")
    template_verts = load_pickle(template_verts_path)
    return template_verts[human_id]


class VocaItem(TypedDict):
    audio: torch.Tensor
    verts: torch.Tensor
    template_vert: torch.Tensor
    one_hot: torch.Tensor
    feature: torch.Tensor


class DataSplitRecorder:
    def __init__(self, write: bool = True) -> None:
        self.write = write
        self.train_list = []
        self.val_list = []
        self.test_list = []

    def write_only(func):
        def wrapper(self, *args, **kwargs):
            if self.write:
                return func(self, *args, **kwargs)
            else:
                raise ValueError("This instance is read only")

        return wrapper

    @write_only
    def add(
        self, humnan_id: str, sentence_id: str, clip_index: int, data_verts_index: int
    ):
        if humnan_id in training_subject and sentence_id in training_sentence:
            self.train_list.append(
                (humnan_id, sentence_id, clip_index, data_verts_index)
            )
        elif humnan_id in validation_subject and sentence_id in validation_sentence:
            self.val_list.append((humnan_id, sentence_id, clip_index, data_verts_index))
        else:
            self.test_list.append(
                (humnan_id, sentence_id, clip_index, data_verts_index)
            )

    @write_only
    def save(self, path: str):
        import pandas as pd

        def _save_list(ls, name):
            df = pd.DataFrame(
                ls,
                columns=["human_id", "sentence_id", "clip_index", "data_verts_index"],
            )
            df.to_csv(name, index=False)

        _save_list(self.train_list, f"{path}/train_list.csv")
        _save_list(self.val_list, f"{path}/val_list.csv")
        _save_list(self.test_list, f"{path}/test_list.csv")

    @staticmethod
    def build(
        raw_audio: VOCASET_AUDIO_TYPE,
        subj_seq_to_idx: VOCASET_WAV_SEQ_TO_IDX_TYPE,
        *,
        save_path: str | None = None,
    ):
        save_path = os.path.join(save_path, "split")
        os.makedirs(save_path, exist_ok=False)
        data_split_recorder = DataSplitRecorder()
        for clip_name, clip_data in raw_audio.items():
            if clip_name not in subj_seq_to_idx:
                continue
            for sentence_id, audio_data in clip_data.items():
                if sentence_id not in subj_seq_to_idx[clip_name]:
                    continue
                audio_idxs_mapping = subj_seq_to_idx[clip_name][sentence_id]
                for clip_index, seq_num in audio_idxs_mapping.items():
                    data_split_recorder.add(clip_name, sentence_id, clip_index, seq_num)

        data_split_recorder.save(save_path)

    @staticmethod
    def exists(path: str):
        path = os.path.join(path, "split")
        return (
            os.path.exists(f"{path}/train_list.csv")
            and os.path.exists(f"{path}/val_list.csv")
            and os.path.exists(f"{path}/test_list.csv")
        )

    @classmethod
    def load(cls, path: str):
        path = os.path.join(path, "split")

        def _load_list(name):
            df = pd.read_csv(name, header=0)
            return df.to_numpy().tolist()

        train_list = _load_list(f"{path}/train_list.csv")
        val_list = _load_list(f"{path}/val_list.csv")
        test_list = _load_list(f"{path}/test_list.csv")
        recorder = cls(write=False)
        recorder.train_list = train_list
        recorder.val_list = val_list
        recorder.test_list = test_list
        return recorder

    def get_list(self, phase: Literal["train", "val", "test", "all"] = "all"):
        if phase == "train":
            return self.train_list
        elif phase == "val":
            return self.val_list
        elif phase == "test":
            return self.test_list
        else:
            return self.train_list + self.val_list + self.test_list


class ClipVocaSet(Dataset):
    def __init__(
        self,
        datapath: str,
        phase: Literal["train", "val", "test", "all"] = "all",
        random_shift: bool = False,
    ):
        self.phase = phase
        self.random_shift = random_shift
        self.datapath = os.path.abspath(datapath)

        self.tempalte_verts: Mapping[str, np.ndarray] = load_pickle(
            os.path.join(self.datapath, "templates.pkl")
        )
        self.raw_audio: VOCASET_AUDIO_TYPE = load_pickle(
            os.path.join(self.datapath, "raw_audio_fixed.pkl")
        )

        self.data_verts: VOCASET_VERTS_TYPE = load_npy_mmapped(
            os.path.join(self.datapath, "data_verts.npy")
        )

        self.wav_seq_to_idx: VOCASET_WAV_SEQ_TO_IDX_TYPE = load_pickle(
            os.path.join(self.datapath, "subj_seq_to_idx.pkl")
        )

        if not DataSplitRecorder.exists(self.datapath):
            print("Building dataset...")
            DataSplitRecorder.build(
                raw_audio=self.raw_audio,
                subj_seq_to_idx=self.wav_seq_to_idx,
                save_path=self.datapath,
            )

        self.split_recorder = DataSplitRecorder.load(self.datapath)
        self.datalist = self.split_recorder.get_list(phase)
        print(f"Loaded {len(self.datalist)} frame data of phase {self.phase}")

    def __repr__(self) -> str:
        return f"""ClipVocaSet(
    phase={self.phase},
    datapath={self.datapath},
    len={len(self.datalist)}
)"""

    def get_single_item(self, key) -> VocaItem:
        human_id, sentence_id, audio_index, data_verts_index = key

        audio = self.raw_audio[human_id][sentence_id]["audio"]
        sr = self.raw_audio[human_id][sentence_id]["sample_rate"]
        verts = self.data_verts[data_verts_index]
        if self.random_shift and self.phase == "train":
            random_shift = random.randint(-500, 500)  # 1%
        else:
            random_shift = 0
        audio_clip = get_audio_fragment(
            audio, audio_index, fps=60, sample_rate=sr, length=0.52, shift=random_shift
        )

        return VocaItem(
            audio=torch.FloatTensor(audio_clip.copy()),
            verts=torch.FloatTensor(verts.copy()),
            template_vert=torch.FloatTensor(self.tempalte_verts[human_id].copy()),
            one_hot=torch.FloatTensor(get_human_id_one_hot(human_id).copy()),
        )

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        return self.get_single_item(self.datalist[idx])

    def get_framedatas(self, target_human_id: str, target_sentence_id: str):
        res = []
        for data_info in self.datalist:
            if data_info[0] == target_human_id and data_info[1] == target_sentence_id:
                res.append(
                    (
                        self.get_single_item(data_info),
                        data_info[2],
                    )
                )
        sorted(res, key=lambda x: x[1])

        return [x[0] for x in res]


class AduioParams(TypedDict):
    fps: int
    sample_rate: int
    length: float
    shift: int


def get_audio_fragment(
    audio: np.ndarray, idx: int, **audio_params: Unpack[AduioParams]
) -> np.ndarray | None:
    l_pad_samples = (
        int(audio_params["sample_rate"] * audio_params["length"] / 2)
        + audio_params["shift"]
    )
    n_pad_samples = int(audio_params["sample_rate"] * audio_params["length"] / 2)
    pad_audio = np.concatenate(
        [np.zeros(l_pad_samples), audio, np.zeros(2 * n_pad_samples)]
    )
    start = idx * audio_params["sample_rate"] // audio_params["fps"]
    end = start + n_pad_samples * 2
    # check if the audio is long enough
    if end > len(pad_audio):
        print(f"Audio is not long enough to get fragment: {end} > {len(pad_audio)}")
        return None
    return pad_audio[start:end]


if __name__ == "__main__":
    dataset = ClipVocaSet("../")
    print(dataset)

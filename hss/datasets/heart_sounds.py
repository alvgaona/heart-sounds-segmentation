import os
from typing import Any
import pandas as pd
import torch
import torchaudio

import pickle

import torchvision.transforms
from torch.utils.data import Dataset
from torch.hub import download_url_to_file
from torchaudio.datasets.utils import _extract_zip as extract_zip

from hss.utils.files import walk_files


def collate_fn(batch):
    batch_size = len(batch)

    if batch_size == 1:
        return batch

    tensors = [torch.reshape(t[0], (-1, 1)) for t in batch]
    padded_tensors = torch.nn.utils.rnn.pad_sequence(tensors)

    for i in range(0, batch_size):
        batch[i] = list(batch[i])
        batch[i][0] = torch.reshape(padded_tensors[:, i], (1, -1))
        batch[i] = tuple(batch[i])

    return batch


class PhysionetChallenge2016(Dataset):
    """ Heart Sounds Dataset from PhysioNet Challenge 2016 """
    _ext_audio = ".wav"
    _ext_reference = ".csv"

    def __init__(
            self,
            root: str,
            url: str = "training",
            train: bool = True,
            download: bool = False,
            transform: torchvision.transforms.Compose = None,
    ):
        """
        Instantiate HeartSoundsAudio object.

        Args:
            root_dir (str): Directory with all the images.
            train (bool, optional): If True, creates dataset from training,
            otherwise from test.pt
            download (bool):
        """
        self.root = root
        self.train = train
        self.transform = transform

        if train is False:
            url = "validation"

        ext_archive = ".zip"
        base_url = "https://www.physionet.org/files/challenge-2016/1.0.0/"
        url = os.path.join(base_url, url + ext_archive)

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.split(".")[0]

        folder_in_archive = basename
        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    download_url_to_file(url + "?download", f"{root}/{archive}")
                    extract_zip(archive, to_path=f"{root}/{basename}")

        walker = walk_files(
            self._path, suffix=self._ext_audio, prefix=True, remove_suffix=True
        )
        self._walker = list(walker)

    def __getitem__(self, n):
        file_id = self._walker[n]
        reference = "REFERENCE"

        path = os.path.dirname(file_id)
        df = pd.read_csv(
            os.path.join(path, reference + self._ext_reference),
            names=["ID", "Condition"]
        )

        set_name = path.split("/")[-1]
        basename = os.path.basename(file_id)

        record = df.loc[df["ID"].isin([basename])]
        label = record.iloc[0]["Condition"]

        file_audio = file_id + self._ext_audio
        output, sample_rate = torchaudio.load(file_audio)

        output = self.transform(output[0])

        return (
            output,
            sample_rate,
            label,
            set_name,
            basename
        )

    def __len__(self) -> int:
        return len(self._walker)

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


class DavidSpringerHSS(Dataset):

    def __init__(
            self,
            dst: str,
            download: bool = False,
            transform: torchvision.transforms.Compose = None,
            dtype: torch.dtype = torch.float64,
    ) -> None:
        self.dst = dst
        self.transform = transform
        self.dtype = dtype

        url = "https://pub-db0cd070a4f94dabb9b58161850d4868.r2.dev/heart-sounds/springer_sounds.zip"
        basename, archive_ext = os.path.basename(url).split('.')

        if download:
            if not os.path.isdir(f"{self.dst}/{basename}"):
                if not os.path.isfile(basename + "." + archive_ext):
                    download_url_to_file(url, f"{dst}/{basename}.{archive_ext}")
                    extract_zip(os.path.join(f"{dst}/{basename}.{archive_ext}"), to_path=dst)
                    os.remove(os.path.join(f"{dst}/{basename}.{archive_ext}"))

        walker = walk_files(
            self.dst, suffix=".pkl", prefix=True, remove_suffix=True
        )
        self.walker = list(walker)

    def __getitem__(self, n) -> Any:
        file_id = self.walker[n]

        with open(file_id + ".pkl", "rb") as f:
            x = pickle.load(f).flatten()

        df = pd.read_csv(file_id + ".csv")

        labels = torch.tensor(df.values.reshape(1, -1)[0], dtype=torch.int8)

        return (
            torch.tensor(x, dtype=self.dtype),
            labels,
        )

    def __len__(self) -> int:
        return len(self.walker)

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)

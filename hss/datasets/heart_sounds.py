import os
from itertools import islice
from typing import Any, Optional

import pandas as pd
import torch
import torchaudio
import torchvision.transforms
from rich.progress import track
from torch.hub import download_url_to_file
from torch.utils.data import Dataset
from torchaudio.datasets.utils import _extract_zip as extract_zip

from hss.transforms import Resample
from hss.utils.files import walk_files
from hss.utils.preprocess import frame_signal


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
    """Heart Sounds Dataset from PhysioNet Challenge 2016"""

    _ext_audio = ".wav"
    _ext_reference = ".csv"

    def __init__(
        self,
        root: str,
        url: str = "training",
        train: bool = True,
        download: bool = False,
        transform: torchvision.transforms.Compose | None = None,
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

        if download and not os.path.isdir(self._path) and not os.path.isfile(archive):
            download_url_to_file(url + "?download", f"{root}/{archive}")
            extract_zip(archive, to_path=f"{root}/{basename}")

        walker = walk_files(self._path, suffix=self._ext_audio, prefix=True, remove_suffix=True)
        self._walker = list(walker)

    def __getitem__(self, n):
        file_id = self._walker[n]
        reference = "REFERENCE"

        path = os.path.dirname(file_id)
        df = pd.read_csv(
            os.path.join(path, reference + self._ext_reference),
            names=["ID", "Condition"],
        )

        set_name = path.split("/")[-1]
        basename = os.path.basename(file_id)

        record = df.loc[df["ID"].isin([basename])]
        label = record.iloc[0]["Condition"]

        file_audio = file_id + self._ext_audio
        output, sample_rate = torchaudio.load(file_audio)

        output = self.transform(output[0]) if self.transform is not None else output[0]

        return output, sample_rate, label, set_name, basename

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
        in_memory: bool = False,
        framing: bool = False,
        stride: int = 1000,
        frame_len: int = 2000,
        count: Optional[int] = None,
        transform: Optional[torchvision.transforms.Compose] = None,
        dtype: torch.dtype = torch.float32,
        verbose: bool = True,
    ) -> None:
        self.dst = dst
        self.transform = transform
        self.dtype = dtype
        self.in_memory = in_memory
        self.data = []

        url = "https://pub-db0cd070a4f94dabb9b58161850d4868.r2.dev/heart-sounds/springer_sounds.zip"
        basename, archive_ext = os.path.basename(url).split(".")

        # The expectation is that the directory path/to/springer_sounds does not exist
        # Also the expectation is that the zip file does not exist, in order to download it
        if (
            download
            and not os.path.isdir(f"{self.dst}/{basename}")
            and not os.path.isfile(basename + "." + archive_ext)
        ):
            download_url_to_file(url, f"{dst}/{basename}.{archive_ext}")
            extract_zip(
                os.path.join(f"{dst}/{basename}.{archive_ext}"),
                to_path=dst,
            )
            os.remove(os.path.join(f"{dst}/{basename}.{archive_ext}"))

        walker = walk_files(self.dst, suffix=".csv", prefix=True, remove_suffix=True)

        if in_memory:
            file_ids = list(walker if not count else islice(walker, count))
            for file_id in track(file_ids, description="Loading Springer dataset...", disable=not verbose):
                x, y = self._load_file(file_id)

                if framing:
                    if len(x) < frame_len:
                        continue

                    frames, labels = frame_signal(x, y - 1, stride, frame_len)

                    for _, (frame, label) in enumerate(zip(frames, labels, strict=False)):
                        frame_i, label_i = self._apply_transform(frame, label)
                        self.data.append((frame_i.to(self.dtype), label_i.squeeze(1)))
                    continue

                self.data.append((x, y))

        self.walker = list(walker)

    def __getitem__(self, n) -> Any:
        if self.in_memory:
            return self.data[n]

        file_id = self.walker[n]
        try:
            x, y = self._load_file(file_id)
            return self._apply_transform(x, y)
        except RuntimeError:
            print(f"Error produced for file {os.path.basename(file_id) + '.csv'}")

    def __len__(self) -> int:
        return len(self.walker) or len(self.data)

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)

    def _load_file(self, file_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        df = pd.read_csv(file_id + ".csv", skiprows=1, names=["Signals", "Labels"])
        x = torch.tensor(df.loc[:, "Signals"].to_numpy(), dtype=self.dtype)
        y = torch.tensor(df.loc[:, "Labels"].to_numpy(), dtype=torch.int64)
        return x, y

    def _apply_transform(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.transform is not None:
            x = self.transform(x)
            for t in self.transform.transforms:
                # Looks for the first resample transform, if there's any
                # to match the length of the new resampled signal.
                if isinstance(t, Resample):
                    y = torch.round(t(y)).type(torch.int64) - 1
                    break

        if len(x.shape) == 1:
            x = x.unsqueeze(1)

        return x, y

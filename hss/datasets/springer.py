import os
from itertools import islice

import pandas as pd
import torch
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


class DavidSpringerHSS(Dataset):
    def __init__(
        self,
        dst: str,
        download: bool = False,
        in_memory: bool = False,
        framing: bool = False,
        stride: int = 1000,
        frame_len: int = 2000,
        count: int | None = None,
        transform: torchvision.transforms.Compose | None = None,
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
        dataset_path = os.path.join(self.dst, basename)

        if download and not os.path.isdir(dataset_path) and not os.path.isfile(basename + "." + archive_ext):
            os.makedirs(dst, exist_ok=True)
            download_url_to_file(url, f"{dst}/{basename}.{archive_ext}")
            extract_zip(
                os.path.join(f"{dst}/{basename}.{archive_ext}"),
                to_path=dst,
            )
            os.remove(os.path.join(f"{dst}/{basename}.{archive_ext}"))

        walker = walk_files(dataset_path, suffix=".csv", prefix=True, remove_suffix=True)

        if in_memory:
            file_ids = list(walker if not count else islice(walker, count))
            for file_id in track(file_ids, description="Loading dataset...", disable=not verbose):
                x, y = self._load_file(file_id)

                if framing:
                    if len(x) < frame_len:
                        continue

                    frames, labels = frame_signal(x, y - 1, stride, frame_len)

                    for frame, label in zip(frames, labels, strict=False):
                        frame_i, label_i = self._apply_transform(frame, label)
                        self.data.append((frame_i.to(self.dtype), label_i.squeeze(1)))
                    continue

                self.data.append((x, y))

        self.walker = list(walker)

    def __getitem__(self, n):
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

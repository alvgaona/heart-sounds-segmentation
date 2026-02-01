from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms
from huggingface_hub import snapshot_download
from rich.progress import track
from torch.utils.data import Dataset

from hss.transforms import Resample
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
    REPO_ID = "alvgaona/heart-sounds"
    FOLDER_NAME = "springer"

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
        self.root = Path(dst)
        self.transform = transform
        self.dtype = dtype
        self.in_memory = in_memory
        self.data: list[tuple[torch.Tensor, torch.Tensor]] = []

        dataset_path = self.root / self.FOLDER_NAME

        if download and not dataset_path.exists():
            self._download()

        # Get sorted list of parquet files
        self.recordings = sorted(dataset_path.glob("*.parquet"))
        if count is not None:
            self.recordings = self.recordings[:count]

        if in_memory:
            for path in track(self.recordings, description="Loading dataset...", disable=not verbose):
                x, y = self._load_recording(path)

                if framing:
                    if len(x) < frame_len:
                        continue

                    frames, labels = frame_signal(x, y - 1, stride, frame_len)

                    for frame, label in zip(frames, labels, strict=False):
                        frame_i, label_i = self._apply_transform(frame, label)
                        self.data.append((frame_i.to(self.dtype), label_i.squeeze(1)))
                    continue

                self.data.append((x, y))

    def __getitem__(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.in_memory:
            return self.data[n]

        path = self.recordings[n]
        x, y = self._load_recording(path)
        return self._apply_transform(x, y)

    def __len__(self) -> int:
        return len(self.data) if self.in_memory else len(self.recordings)

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)

    def _download(self) -> None:
        """Download the dataset from HuggingFace Hub."""
        snapshot_download(
            repo_id=self.REPO_ID,
            repo_type="dataset",
            local_dir=self.root,
            allow_patterns=f"{self.FOLDER_NAME}/*",
        )

    def _load_recording(self, path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        df = pd.read_parquet(path)
        x = torch.tensor(df["signals"].to_numpy(), dtype=self.dtype)
        y = torch.tensor(df["labels"].to_numpy(), dtype=torch.int64)
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

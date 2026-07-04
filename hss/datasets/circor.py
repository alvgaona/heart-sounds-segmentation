from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal
import torch
import torchvision.transforms
from huggingface_hub import snapshot_download
from rich.progress import track
from torch.utils.data import Dataset

from hss.utils.preprocess import frame_signal


def _resample_signal(x: torch.Tensor, target_len: int) -> torch.Tensor:
    return torch.tensor(scipy.signal.resample(x.numpy(), target_len), dtype=x.dtype)


def _resample_labels_nn(y: torch.Tensor, target_len: int) -> torch.Tensor:
    """Nearest-neighbour label resample — never interpolates across state boundaries or the -1 sentinel."""
    src = y.numpy()
    idx = np.minimum((np.arange(target_len) * len(src) / target_len).round().astype(int), len(src) - 1)
    return torch.tensor(src[idx], dtype=torch.int64)


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


class CirCorDataset(Dataset):
    """CirCor DigiScope Phonocardiogram Dataset (PhysioNet Challenge 2022).

    Parquets are the official .wav cropped to the annotated span (interior unannotated regions kept, leading/
    trailing dropped) at 4000 Hz, with per-sample labels 1=S1, 2=Systole, 3=S2, 4=Diastole and -1 = unannotated.

    This loader resamples each recording to ``target_fs`` (default 1000 Hz, matching the Springer pipeline) and
    maps labels to the model's 0-indexed space: 1-4 -> 0-3, with -1 preserved as the ignore label. Signals are
    Fourier-resampled; labels are nearest-neighbour resampled so state boundaries and the -1 sentinel are never
    interpolated across.
    """

    REPO_ID = "alvgaona/heart-sounds"
    FOLDER_NAME = "circor"

    def __init__(
        self,
        root: str,
        download: bool = False,
        in_memory: bool = False,
        framing: bool = False,
        stride: int = 1000,
        frame_len: int = 2000,
        count: int | None = None,
        transform: torchvision.transforms.Compose | None = None,
        dtype: torch.dtype = torch.float32,
        verbose: bool = True,
        orig_fs: int = 4000,
        target_fs: int = 1000,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.dtype = dtype
        self.in_memory = in_memory
        self.orig_fs = orig_fs
        self.target_fs = target_fs
        self.data: list[tuple[torch.Tensor, torch.Tensor]] = []

        dataset_path = self.root / self.FOLDER_NAME

        if download and not dataset_path.exists():
            self._download()

        # Get sorted list of parquet files (exclude metadata.parquet)
        self.recordings = sorted(p for p in dataset_path.glob("*.parquet") if p.name != "metadata.parquet")
        if count is not None:
            self.recordings = self.recordings[:count]

        if in_memory:
            for path in track(self.recordings, description="Loading CirCor dataset...", disable=not verbose):
                x, y = self._load_recording(path)

                if framing:
                    if len(x) < frame_len:
                        continue
                    frames, labels = frame_signal(x, y, stride, frame_len)
                    for frame, label in zip(frames, labels, strict=False):
                        frame_t, label_t = self._apply_transform(frame, label)
                        self.data.append((frame_t.to(self.dtype), label_t.squeeze(1)))
                else:
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
        """Load a parquet, resample to target_fs, and map labels to 0-3 with -1 preserved."""
        df = pd.read_parquet(path)
        x = torch.tensor(df["signals"].to_numpy(), dtype=self.dtype)
        y = torch.tensor(df["labels"].to_numpy(), dtype=torch.int64)

        if self.target_fs != self.orig_fs:
            target_len = round(len(x) * self.target_fs / self.orig_fs)
            x = _resample_signal(x, target_len)
            y = _resample_labels_nn(y, target_len)

        # CirCor labels 1-4 -> model space 0-3; -1 (unannotated) stays -1 as the ignore label.
        y = torch.where(y > 0, y - 1, y)
        return x, y

    def _apply_transform(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.transform is not None:
            x = self.transform(x)

        if len(x.shape) == 1:
            x = x.unsqueeze(1)

        return x, y

    def get_metadata(self) -> pd.DataFrame:
        """Load and return the metadata DataFrame."""
        metadata_path = self.root / self.FOLDER_NAME / "metadata.parquet"
        return pd.read_parquet(metadata_path)

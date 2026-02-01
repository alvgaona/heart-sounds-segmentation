from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchvision.transforms
from rich.progress import track
from torch.hub import download_url_to_file
from torch.utils.data import Dataset
from torchaudio.datasets.utils import _extract_zip as extract_zip

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


class CirCorDataset(Dataset):
    """CirCor DigiScope Phonocardiogram Dataset (PhysioNet Challenge 2022).

    Loads heart sound recordings with per-sample segmentation labels.
    Labels: 0=S1, 1=Systole, 2=S2, 3=Diastole (0-indexed, matching DavidSpringerHSS).
    Segments with state 0 (unannotated) in the original TSV are excluded.
    """

    SAMPLE_RATE = 4000
    URL = "https://physionet.org/static/published-projects/circor-heart-sound/the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip"
    ARCHIVE_FOLDER = "the-circor-digiscope-phonocardiogram-dataset-1.0.3"
    FOLDER_NAME = "circor_sounds"

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
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.dtype = dtype
        self.in_memory = in_memory
        self.framing = framing
        self.stride = stride
        self.frame_len = frame_len
        self.data: list[tuple[torch.Tensor, torch.Tensor]] = []

        dataset_path = self.root / self.FOLDER_NAME
        training_data = dataset_path / "training_data"

        if download and not training_data.exists():
            self._download()

        wav_files = sorted(training_data.glob("*.wav"))

        if count is not None:
            wav_files = wav_files[:count]

        self.recordings: list[Path] = []

        if in_memory:
            for wav_path in track(wav_files, description="Loading CirCor dataset...", disable=not verbose):
                result = self._load_recording(wav_path)
                if result is None:
                    continue

                x, y = result

                if framing:
                    if len(x) < frame_len:
                        continue
                    frames, labels = frame_signal(x, y, stride, frame_len)
                    for frame, label in zip(frames, labels, strict=False):
                        frame_t, label_t = self._apply_transform(frame, label)
                        self.data.append((frame_t.to(self.dtype), label_t.squeeze(1)))
                else:
                    self.data.append((x, y))
        else:
            self.recordings = wav_files

    def __getitem__(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.in_memory:
            return self.data[n]

        wav_path = self.recordings[n]
        result = self._load_recording(wav_path)
        if result is None:
            raise RuntimeError(f"Failed to load recording: {wav_path}")

        x, y = result
        return self._apply_transform(x, y)

    def __len__(self) -> int:
        return len(self.data) if self.in_memory else len(self.recordings)

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)

    def _load_recording(self, wav_path: Path) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Load a WAV file and its corresponding TSV annotations."""
        tsv_path = wav_path.with_suffix(".tsv")
        if not tsv_path.exists():
            return None

        waveform, sr = torchaudio.load(wav_path)
        if sr != self.SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, self.SAMPLE_RATE)

        audio = waveform.squeeze(0)
        num_samples = len(audio)

        labels = self._parse_tsv(tsv_path, num_samples)
        if labels is None:
            return None

        valid_mask = labels >= 0
        if not valid_mask.any():
            return None

        valid_indices = torch.where(valid_mask)[0]
        start_idx = valid_indices[0].item()
        end_idx = valid_indices[-1].item() + 1

        audio = audio[start_idx:end_idx]
        labels = labels[start_idx:end_idx]

        return audio.to(self.dtype), labels

    def _parse_tsv(self, tsv_path: Path, num_samples: int) -> torch.Tensor | None:
        """Parse TSV annotation file and convert to per-sample labels.

        TSV format: start_time, end_time, state (tab-separated)
        States in file: 0=unannotated, 1=S1, 2=Systole, 3=S2, 4=Diastole
        Output labels: 0=S1, 1=Systole, 2=S2, 3=Diastole, -1=unannotated
        """
        df = pd.read_csv(tsv_path, sep="\t", header=None, names=["start", "end", "state"])

        labels = np.full(num_samples, -1, dtype=np.int64)

        for _, row in df.iterrows():
            start_sample = int(row["start"] * self.SAMPLE_RATE)
            end_sample = int(row["end"] * self.SAMPLE_RATE)
            state = int(row["state"])

            start_sample = max(0, start_sample)
            end_sample = min(num_samples, end_sample)

            if state == 0:
                continue
            labels[start_sample:end_sample] = state - 1

        return torch.from_numpy(labels)

    def _apply_transform(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.transform is not None:
            x = self.transform(x)
            for t in self.transform.transforms:
                if isinstance(t, Resample):
                    y = torch.round(t(y.float())).to(torch.int64)
                    break

        if len(x.shape) == 1:
            x = x.unsqueeze(1)

        return x, y

    def _download(self) -> None:
        """Download and extract the CirCor dataset from PhysioNet."""
        self.root.mkdir(parents=True, exist_ok=True)
        archive_path = self.root / f"{self.ARCHIVE_FOLDER}.zip"

        if not archive_path.exists():
            print(f"Downloading CirCor dataset from {self.URL}...")
            download_url_to_file(self.URL, str(archive_path))

        print("Extracting dataset...")
        extract_zip(str(archive_path), to_path=str(self.root))

        extracted_path = self.root / self.ARCHIVE_FOLDER
        target_path = self.root / self.FOLDER_NAME
        extracted_path.rename(target_path)

        archive_path.unlink()

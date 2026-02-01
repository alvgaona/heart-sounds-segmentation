#!/usr/bin/env python3
"""Train heart sound segmentation model with Semi-Markov CRF (duration-aware)."""

import os

# Set your HuggingFace token here for Colab
HF_TOKEN = None  # e.g., "hf_xxxxx"

import lightning.pytorch as pl
import scipy
import torch
import torch.utils.data
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar
from torch.utils.data import DataLoader
from torchvision import transforms

from hss.datasets import DavidSpringerHSS
from hss.model.lit_model_semi_crf import LitModelSemiCRF
from hss.transforms import FSST


def get_device() -> tuple[torch.device, str]:
    """Get the best available device and accelerator."""
    if torch.cuda.is_available():
        return torch.device("cuda"), "gpu"
    elif torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def main() -> None:
    device, accelerator = get_device()
    print(f"Using device: {device} (accelerator: {accelerator})")
    print("Training with Semi-Markov CRF model (duration-aware)")

    transform = transforms.Compose(
        (
            FSST(
                1000,
                window=scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=False),
                truncate_freq=(25, 200),
                stack=True,
            ),
        )
    )

    hss_dataset = DavidSpringerHSS(
        "data",
        download=True,
        framing=True,
        in_memory=True,
        transform=transform,
        token=HF_TOKEN,
    )

    batch_size = 50

    # Simple train/val/test split (no k-fold for quick comparison)
    test_size = int(0.15 * len(hss_dataset))
    val_size = int(0.15 * len(hss_dataset))
    train_size = len(hss_dataset) - test_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        hss_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(68)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count() or 4,
        drop_last=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count() or 4,
        drop_last=True,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count() or 4,
        drop_last=True,
    )

    # Initialize Semi-Markov CRF model with duration priors
    # At 50 Hz (FSST output rate), durations in frames:
    # S1: ~100-150ms -> 5-8 frames
    # Systole: ~200-300ms -> 10-15 frames
    # S2: ~80-120ms -> 4-6 frames
    # Diastole: ~300-500ms -> 15-25 frames (variable with heart rate)
    model = LitModelSemiCRF(
        input_size=44,
        batch_size=batch_size,
        device=device,
        max_duration=100,  # Max 2 seconds at 50 Hz
        duration_means=[6.0, 12.0, 5.0, 20.0],  # Initial estimates
        duration_stds=[2.0, 4.0, 2.0, 8.0],
    )

    early_stopping = EarlyStopping("val_loss", patience=6, check_finite=True)

    trainer = pl.Trainer(
        max_epochs=15,
        accelerator=accelerator,
        gradient_clip_val=1,
        gradient_clip_algorithm="norm",
        callbacks=[early_stopping, RichProgressBar()],
        default_root_dir="lightning_logs_semi_crf",
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Test
    test_results = trainer.test(dataloaders=test_loader, ckpt_path="best")[0]

    print("\n" + "=" * 60)
    print("SEMI-MARKOV CRF MODEL TEST RESULTS")
    print("=" * 60)
    for key, value in sorted(test_results.items()):
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()

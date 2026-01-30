#!/usr/bin/env python3
"""Train heart sound segmentation model with attention mechanism."""

import os

import lightning.pytorch as pl
import scipy
import torch
import torch.utils.data
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar
from torch.utils.data import DataLoader
from torchvision import transforms

from hss.datasets.heart_sounds import DavidSpringerHSS
from hss.model.lit_model_attention import LitModelAttention
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
    print("Training with LSTM + Attention model")

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
        "resources/data",
        download=True,
        framing=True,
        in_memory=True,
        transform=transform,
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

    # Initialize attention model
    model = LitModelAttention(
        input_size=44,
        batch_size=batch_size,
        device=device,
        num_attention_heads=8,
    )

    early_stopping = EarlyStopping("val_loss", patience=6, check_finite=True)

    trainer = pl.Trainer(
        max_epochs=15,
        accelerator=accelerator,
        gradient_clip_val=1,
        gradient_clip_algorithm="norm",
        callbacks=[early_stopping, RichProgressBar()],
        default_root_dir="lightning_logs_attention",
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Test
    test_results = trainer.test(dataloaders=test_loader, ckpt_path="best")[0]

    print("\n" + "=" * 60)
    print("ATTENTION MODEL TEST RESULTS")
    print("=" * 60)
    for key, value in sorted(test_results.items()):
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()

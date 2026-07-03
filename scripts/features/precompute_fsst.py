#!/usr/bin/env python3
"""Precompute FSST features for Springer dataset to speed up training."""

from pathlib import Path

import scipy
import torch
from torchvision import transforms

from hss.datasets import DavidSpringerHSS
from hss.transforms import FSST


# Set your HuggingFace token here for Colab
HF_TOKEN = None  # e.g., "hf_xxxxx"

# Output path for precomputed features
OUTPUT_PATH = Path("data/springer_fsst")


def main() -> None:
    print("Loading Springer dataset...")

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

    dataset = DavidSpringerHSS(
        "data",
        download=True,
        framing=True,
        in_memory=True,
        transform=transform,
        token=HF_TOKEN,
    )

    print(f"Dataset size: {len(dataset)} frames")

    # Create output directory
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Save all frames as a single tensor file
    features = []
    labels = []

    for i, (x, y) in enumerate(dataset):
        features.append(x)
        labels.append(y)

    features = torch.stack(features)
    labels = torch.stack(labels)

    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    torch.save({"features": features, "labels": labels}, OUTPUT_PATH / "springer_fsst.pt")
    print(f"Saved to {OUTPUT_PATH / 'springer_fsst.pt'}")


if __name__ == "__main__":
    main()

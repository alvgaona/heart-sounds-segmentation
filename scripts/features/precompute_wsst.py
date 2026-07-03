#!/usr/bin/env python3
"""Precompute WSST features for the Springer dataset to speed up training.

Mirror of precompute_fsst.py using the Wavelet Synchrosqueezed Transform (ssq.wsst). The feature dimension
is 2 * (number of log-spaced bins inside --truncate-freq), which grows with --num-voices. Full resolution
(--downsample 1) keeps the same code path as the FSST features (train with --downsample 20); pre-pooling
here (--downsample 20) caps RAM/disk for high --num-voices and is trained with --downsample 1.
"""

import argparse
from pathlib import Path

import torch
from torchvision import transforms

from hss.datasets import DavidSpringerHSS
from hss.transforms import WSST


# Set your HuggingFace token here for Colab
HF_TOKEN = None  # e.g., "hf_xxxxx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wavelet", choices=["amor", "bump"], default="amor", help="Mother wavelet")
    parser.add_argument("--num-voices", type=int, default=8, help="Voices per octave (sets feature dimension)")
    parser.add_argument(
        "--truncate-freq",
        type=float,
        nargs=2,
        default=(25.0, 200.0),
        metavar=("FMIN", "FMAX"),
        help="Frequency band (Hz) kept before stacking real/imag",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Avg-pool factor applied to features (labels majority-voted) before saving. 1 keeps full 1000 Hz "
        "(train with --downsample 20); 20 pre-pools to 50 Hz (train with --downsample 1).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default data/springer_wsst_nv{num_voices})",
    )
    return parser.parse_args()


def downsample_time(features: torch.Tensor, labels: torch.Tensor, factor: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Average-pool features and majority-vote labels along time; drops the partial tail window."""
    if factor <= 1:
        return features, labels
    n, t, c = features.shape
    t2 = (t // factor) * factor
    new_t = t2 // factor
    features = features[:, :t2, :].reshape(n, new_t, factor, c).mean(dim=2)
    labels = labels[:, :t2].reshape(n, new_t, factor).mode(dim=2).values
    return features, labels


def main(args: argparse.Namespace) -> None:
    out_path = args.out or Path(f"data/springer_wsst_nv{args.num_voices}")
    print(f"Loading Springer dataset (wavelet={args.wavelet}, num_voices={args.num_voices})...")

    transform = transforms.Compose(
        (
            WSST(
                1000,
                wavelet=args.wavelet,
                num_voices=args.num_voices,
                truncate_freq=tuple(args.truncate_freq),
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
    out_path.mkdir(parents=True, exist_ok=True)

    features = []
    labels = []
    for x, y in dataset:
        features.append(x)
        labels.append(y)

    features = torch.stack(features)
    labels = torch.stack(labels)

    if args.downsample > 1:
        features, labels = downsample_time(features, labels, args.downsample)
        print(f"Downsampled x{args.downsample} before saving")

    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    torch.save({"features": features, "labels": labels}, out_path / "springer_wsst.pt")
    print(f"Saved to {out_path / 'springer_wsst.pt'}")


if __name__ == "__main__":
    main(parse_args())

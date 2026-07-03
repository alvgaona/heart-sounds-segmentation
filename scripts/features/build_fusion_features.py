#!/usr/bin/env python3
"""Build a fused feature set by concatenating two precomputed feature .pt files along the channel dim.

Default: FSST-44 (data/springer_fsst) + WSST-nv8-48 (data/springer_wsst_nv8) -> 92 features. Both inputs must
be full-resolution and frame-aligned (identical labels); the two are pooled to the requested rate and then
concatenated. Onset F1 later references the shared 1000 Hz labels, so pre-pooling here is safe.
"""

import argparse
from pathlib import Path

import torch


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", default="data/springer_fsst/springer_fsst.pt", help="First feature .pt (full-res)")
    parser.add_argument("--b", default="data/springer_wsst_nv8/springer_wsst.pt", help="Second feature .pt (full-res)")
    parser.add_argument(
        "--downsample", type=int, default=20, help="Avg-pool factor applied before saving (20 -> 50 Hz)"
    )
    parser.add_argument("--out", type=Path, default=Path("data/springer_fusion"), help="Output directory")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    da = torch.load(args.a, weights_only=True)
    db = torch.load(args.b, weights_only=True)
    fa, la = da["features"], da["labels"]
    fb, lb = db["features"], db["labels"]

    if not torch.equal(la, lb):
        raise ValueError("Label mismatch between inputs; cannot fuse (frame alignment broken).")
    if fa.shape[:2] != fb.shape[:2]:
        raise ValueError(f"Shape mismatch: {tuple(fa.shape)} vs {tuple(fb.shape)}")

    fa, labels = downsample_time(fa, la, args.downsample)
    fb, _ = downsample_time(fb, lb, args.downsample)
    features = torch.cat([fa, fb], dim=-1)

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Fusion features: {tuple(features.shape)} (= {fa.shape[-1]} + {fb.shape[-1]}), labels {tuple(labels.shape)}")
    torch.save({"features": features, "labels": labels}, args.out / "springer_fusion.pt")
    print(f"Saved to {args.out / 'springer_fusion.pt'}")


if __name__ == "__main__":
    main(parse_args())

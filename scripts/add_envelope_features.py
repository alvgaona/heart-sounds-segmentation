#!/usr/bin/env python3
"""Augment the precomputed FSST .pt with homomorphic + Hilbert envelope channels (44 -> 46). No FSST recompute.

The raw waveforms are re-framed from the dataset in the same deterministic order as precompute_fsst.py
(sorted recordings, frame_signal(x, y-1, stride=1000, len=2000)), so raw frame k aligns 1:1 to FSST frame k
(asserted via matching labels). Envelopes are z-normalised per frame then scaled to the FSST magnitude so
the new channels sit on the same scale as the existing features. See hss/transforms/envelope.py.
"""

import argparse

import torch
from rich.progress import track

from hss.datasets import DavidSpringerHSS
from hss.transforms.envelope import envelope_features


FS = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsst-path", default="data/springer_fsst/springer_fsst.pt")
    parser.add_argument("--out-path", default="data/springer_fsst/springer_fsst_env.pt")
    parser.add_argument("--data-root", default="data")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    data = torch.load(args.fsst_path, weights_only=True)
    features, labels = data["features"], data["labels"]
    n, t, c = features.shape
    print(f"FSST features: {tuple(features.shape)}  labels: {tuple(labels.shape)}")

    print("Loading raw waveforms (re-framing the dataset)...")
    raw = DavidSpringerHSS(args.data_root, download=False, in_memory=True, framing=True, transform=None)
    assert len(raw.data) == n, f"raw {len(raw.data)} != fsst {n} (alignment broken)"
    for k in (0, n // 2, n - 1):
        assert torch.equal(raw.data[k][1], labels[k]), f"label mismatch at frame {k}"
    print(f"Aligned {n} raw frames to FSST frames (labels match).")

    scale = float(features.std())
    print(f"FSST global std = {scale:.4f} (envelope channels rescaled to match)")

    out = torch.empty((n, t, c + 2), dtype=torch.float32)
    out[:, :, :c] = features
    for k in track(range(n), description="Computing envelopes..."):
        env = envelope_features(raw.data[k][0].numpy(), FS)  # (t, 2), z-normed
        out[k, :, c:] = torch.from_numpy(env) * scale

    env_std = float(out[:, :, c:].std())
    print(f"envelope channels std = {env_std:.4f} (target {scale:.4f})")
    print(f"Saving {tuple(out.shape)} -> {args.out_path}")
    torch.save({"features": out, "labels": labels}, args.out_path)
    print("Done.")


if __name__ == "__main__":
    main(parse_args())

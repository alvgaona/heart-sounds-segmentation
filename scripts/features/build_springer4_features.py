#!/usr/bin/env python3
"""Build the standalone Springer-4 feature file (no FSST): homomorphic + Hilbert + wavelet + PSD.

For the FSST-vs-Springer comparison — same frames, labels, folds, and model as the FSST pipeline, so
the only difference is the feature front-end. Raw frames are re-framed from the dataset in the same
deterministic order as precompute_fsst.py, so frame k aligns 1:1 to the FSST .pt (asserted via labels).
"""

import argparse

import torch
from rich.progress import track

from hss.datasets import DavidSpringerHSS
from hss.transforms.envelope import springer_features


FS = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsst-path", default="data/springer_fsst/springer_fsst.pt", help="For labels + alignment")
    parser.add_argument("--out-path", default="data/springer_fsst/springer4.pt")
    parser.add_argument("--data-root", default="data")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    labels = torch.load(args.fsst_path, weights_only=True)["labels"]
    n, t = labels.shape
    print(f"labels: {tuple(labels.shape)}")

    print("Loading raw waveforms (re-framing the dataset)...")
    raw = DavidSpringerHSS(args.data_root, download=False, in_memory=True, framing=True, transform=None)
    assert len(raw.data) == n, f"raw {len(raw.data)} != fsst {n} (alignment broken)"
    for k in (0, n // 2, n - 1):
        assert torch.equal(raw.data[k][1], labels[k]), f"label mismatch at frame {k}"
    print(f"Aligned {n} raw frames (labels match). Computing Springer-4 features...")

    out = torch.empty((n, t, 4), dtype=torch.float32)
    for k in track(range(n), description="Springer features..."):
        out[k] = torch.from_numpy(springer_features(raw.data[k][0].numpy(), FS))

    print(f"features std per channel: {out.reshape(-1, 4).std(dim=0).tolist()}")
    print(f"Saving {tuple(out.shape)} -> {args.out_path}")
    torch.save({"features": out, "labels": labels}, args.out_path)
    print("Done.")


if __name__ == "__main__":
    main(parse_args())

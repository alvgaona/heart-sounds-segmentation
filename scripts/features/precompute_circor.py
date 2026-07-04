#!/usr/bin/env python3
"""Precompute FSST/WSST features for the CirCor dataset (cross-dataset test set for C7).

Mirrors precompute_fsst/precompute_wsst but for CirCor: each recording is resampled 4000->1000 Hz and framed
into 2 s windows (2000 samples, 1 s stride) exactly like Springer, so Springer-trained checkpoints transfer
directly. Unlike the Springer precompute, this ALSO tracks per-frame recording_id / patient_id (for per-recording
and per-patient metric aggregation) and carries 1000 Hz labels (`labels_hr`) for onset-F1 references.

Features are pre-pooled to `--downsample`-Hz (default 50 Hz = factor 20) to keep the file small (~0.6 GB vs
~13 GB at 1000 Hz); the models run at 50 Hz so nothing is lost. Windows that are 100% unannotated (all -1) are
dropped; partial-`-1` windows are kept (mask = label != -1 downstream). Labels are the model's 0-3 space with
-1 = ignore (see CirCorDataset).

Output dict (data/circor_{fsst,wsst_nv{V}}/circor_{fsst,wsst}.pt):
  features     (N, T, C)   pooled to --downsample-Hz
  labels       (N, T)      pooled (majority vote), -1 preserved
  labels_hr    (N, 2000)   1000 Hz per-frame labels for onset-F1 reference
  recording_id (N,)        contiguous index over contributing recordings
  patient_id   (N,)        contiguous index over patients (from metadata.parquet)
  rec_names    list[str]   recording_id string per contiguous recording index
"""

import argparse
from pathlib import Path

import pandas as pd
import scipy.signal
import torch
from rich.progress import track
from torchvision import transforms

from hss.datasets import CirCorDataset
from hss.transforms import FSST, WSST
from hss.utils.preprocess import frame_signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", choices=["fsst", "wsst"], default="wsst")
    parser.add_argument("--wavelet", choices=["amor", "bump"], default="amor")
    parser.add_argument("--num-voices", type=int, default=8, help="WSST voices per octave (feature dim)")
    parser.add_argument("--truncate-freq", type=float, nargs=2, default=(25.0, 200.0), metavar=("FMIN", "FMAX"))
    parser.add_argument("--dataset-path", default="data")
    parser.add_argument("--frame-len", type=int, default=2000, help="Window length at 1000 Hz (2 s)")
    parser.add_argument("--stride", type=int, default=1000, help="Window stride at 1000 Hz (1 s)")
    parser.add_argument("--downsample", type=int, default=20, help="Avg-pool factor for saved features (20 = 50 Hz)")
    parser.add_argument("--count", type=int, default=None, help="Cap recordings (smoke test)")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def build_transform(args: argparse.Namespace):
    if args.features == "fsst":
        return transforms.Compose(
            (
                FSST(
                    1000,
                    window=scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=False),
                    truncate_freq=tuple(args.truncate_freq),
                    stack=True,
                ),
            )
        )
    return transforms.Compose(
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


def pool_feat(f: torch.Tensor, factor: int) -> torch.Tensor:
    if factor <= 1:
        return f
    t2 = (f.shape[0] // factor) * factor
    return f[:t2].reshape(t2 // factor, factor, -1).mean(dim=1)


def pool_lab(lab: torch.Tensor, factor: int) -> torch.Tensor:
    if factor <= 1:
        return lab
    t2 = (len(lab) // factor) * factor
    return lab[:t2].reshape(t2 // factor, factor).mode(dim=1).values


def main(args: argparse.Namespace) -> None:
    default_dir = "circor_fsst" if args.features == "fsst" else f"circor_wsst_nv{args.num_voices}"
    out_path = args.out or Path(f"data/{default_dir}")
    out_path.mkdir(parents=True, exist_ok=True)
    fname = "circor_fsst.pt" if args.features == "fsst" else "circor_wsst.pt"

    ds = CirCorDataset(args.dataset_path, download=False, in_memory=False, transform=build_transform(args))
    if args.count is not None:
        ds.recordings = ds.recordings[: args.count]

    meta = pd.read_parquet(Path(args.dataset_path) / "circor" / "metadata.parquet")
    rec2pat = dict(zip(meta["recording_id"], meta["patient_id"], strict=True))
    patients = sorted(meta["patient_id"].unique().tolist())
    pat2idx = {p: i for i, p in enumerate(patients)}
    print(f"CirCor: {len(ds.recordings)} recordings, {len(patients)} patients | features={args.features}")

    features, labels, labels_hr, rec_ids, pat_ids, rec_names = [], [], [], [], [], []
    rec_idx = 0
    dropped = 0
    for path in track(ds.recordings, description="Precomputing CirCor features..."):
        x, y = ds._load_recording(path)  # 1000 Hz signal, labels 0-3 / -1
        if len(x) < args.frame_len:
            continue
        frames, frame_labels = frame_signal(x, y, args.stride, args.frame_len)
        rid = path.stem
        pidx = pat2idx[rec2pat[rid]]
        used = False
        for frame, lab in zip(frames, frame_labels, strict=False):
            lab = lab.squeeze(1) if lab.dim() > 1 else lab
            if bool((lab == -1).all()):  # window fully unannotated -> useless
                dropped += 1
                continue
            feat = ds._apply_transform(frame, lab)[0]
            features.append(pool_feat(feat, args.downsample).to(torch.float32))
            labels.append(pool_lab(lab, args.downsample))
            labels_hr.append(lab.to(torch.int8))
            rec_ids.append(rec_idx)
            pat_ids.append(pidx)
            used = True
        if used:
            rec_names.append(rid)
            rec_idx += 1

    out = {
        "features": torch.stack(features),
        "labels": torch.stack(labels),
        "labels_hr": torch.stack(labels_hr),
        "recording_id": torch.tensor(rec_ids, dtype=torch.long),
        "patient_id": torch.tensor(pat_ids, dtype=torch.long),
        "rec_names": rec_names,
    }
    print(
        f"frames={out['features'].shape[0]} feat={tuple(out['features'].shape)} "
        f"recordings={rec_idx} patients={len(set(pat_ids))} dropped_all_-1={dropped}"
    )
    torch.save(out, out_path / fname)
    print(f"Saved to {out_path / fname}")


if __name__ == "__main__":
    main(parse_args())

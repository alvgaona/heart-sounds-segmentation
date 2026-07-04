#!/usr/bin/env python3
"""Build frame->recording (and optionally frame->patient) indices for grouped CV (additive; nothing mutated).

The precomputed feature .pt is a flat stack of frames with no grouping, so recording/patient-level splits need
to know which recording/patient each frame came from. This reproduces the dataset's deterministic framing
(sorted recordings, `frame_signal(stride, frame_len)`, sub-frame_len recordings skipped) and emits a length-N
frame->recording tensor. Feature-independent (same for FSST/WSST), so one file serves all.

Patient index: pass --patient-mat pointing at the PhysioNet HSS `example_data.mat`
(https://physionet.org/content/hss/1.0/, field `example_data.patient_number`, 792 recordings, 135 patients).
The parquet order aligns to `example_data` exactly (verified: signals bit-identical), so patient_number[j]
attaches to sorted recording j. Springer 2016 split by patient, so this enables the matching protocol.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from hss.datasets import DavidSpringerHSS
from hss.utils.preprocess import frame_signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default="data")
    parser.add_argument("--frame-len", type=int, default=2000)
    parser.add_argument("--stride", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("data/recording_ids.pt"))
    parser.add_argument("--patient-mat", type=Path, default=None, help="PhysioNet HSS example_data.mat (optional)")
    parser.add_argument("--patient-out", type=Path, default=Path("data/patient_ids.pt"))
    parser.add_argument("--expect-frames", type=int, default=None, help="Assert total frames (e.g. 8408) if set")
    return parser.parse_args()


def load_patient_numbers(mat_path: Path) -> np.ndarray:
    import scipy.io as sio

    ed = sio.loadmat(str(mat_path), simplify_cells=True)["example_data"]
    return np.array(ed["patient_number"]).ravel()


def main(args: argparse.Namespace) -> None:
    dataset = DavidSpringerHSS(args.dataset_path, download=False, framing=False, in_memory=True, verbose=False)
    patient_number = load_patient_numbers(args.patient_mat) if args.patient_mat else None
    if patient_number is not None and len(patient_number) != len(dataset):
        raise ValueError(f"patient_number has {len(patient_number)} entries, dataset has {len(dataset)} recordings")

    rec_ids: list[int] = []
    patient_ids: list[int] = []
    rec = 0
    for j, (x, y) in enumerate(dataset):
        if len(x) < args.frame_len:
            continue
        n_frames = len(frame_signal(x, y - 1, args.stride, args.frame_len)[0])
        rec_ids.extend([rec] * n_frames)
        if patient_number is not None:
            patient_ids.extend([int(patient_number[j])] * n_frames)
        rec += 1

    index = torch.tensor(rec_ids, dtype=torch.long)
    if args.expect_frames is not None and len(index) != args.expect_frames:
        raise ValueError(f"built {len(index)} frames, expected {args.expect_frames}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(index, args.out)
    print(f"recordings={rec}, frames={len(index)} -> {args.out}")

    if patient_number is not None:
        pindex = torch.tensor(patient_ids, dtype=torch.long)
        torch.save(pindex, args.patient_out)
        print(f"patients={len(torch.unique(pindex))}, frames={len(pindex)} -> {args.patient_out}")


if __name__ == "__main__":
    main(parse_args())

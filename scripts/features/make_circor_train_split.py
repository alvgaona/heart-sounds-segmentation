#!/usr/bin/env python3
"""Derive a CirCor TRAINING file (0-3 labels, no -1) from a precompute_circor.py output — no recompute.

For training on CirCor (the in-domain control for C7), the CRF/softmax pipeline can't consume -1. CirCor is
99.5% annotated, so we simply keep only fully-annotated windows (every 1000 Hz label != -1); the surviving
50 Hz labels are then pure 0-3 and the existing train_crf.py / train_lstm.py work unchanged (feed the output as
--fsst-path with --downsample 1, and the emitted patient index as --patient-index --split-by patient).

Emits: <out> = {features, labels} (train-ready) and <patient-out> = frame->patient tensor (contiguous patient
indices from the precompute). The keep-mask is on `labels_hr`, identical across FSST/WSST precomputes, so the
patient index matches both.
"""

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", required=True, help="precompute_circor.py output (.pt)")
    parser.add_argument("--out", required=True, help="training feature file to write (.pt)")
    parser.add_argument("--patient-out", default="data/circor_patient_ids.pt", help="frame->patient index (.pt)")
    parser.add_argument(
        "--ref-out",
        default=None,
        help="Optional {labels: 1000 Hz labels_hr of kept windows} for onset-F1 references when re-evaluating "
        "the in-domain checkpoints with reeval_springer_metrics.py (--ref-labels-path).",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    d = torch.load(args.inp, weights_only=False)
    keep = (d["labels_hr"] != -1).all(dim=1)  # fully-annotated windows only
    feats, labs, pat = d["features"][keep], d["labels"][keep], d["patient_id"][keep]
    if int(labs.min()) < 0:
        raise ValueError("filtered labels still contain -1; keep-mask is wrong")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": feats, "labels": labs}, args.out)
    torch.save(pat, args.patient_out)
    if args.ref_out:
        torch.save({"labels": d["labels_hr"][keep].long()}, args.ref_out)
        print(f"onset ref (1000 Hz) -> {args.ref_out}")
    print(
        f"kept {int(keep.sum())}/{len(keep)} windows ({keep.float().mean():.1%}); dropped {int((~keep).sum())} "
        f"with any -1 | patients {len(torch.unique(pat))} | labels {sorted(torch.unique(labs).tolist())}"
    )
    print(f"features -> {args.out}\npatient index -> {args.patient_out}")


if __name__ == "__main__":
    main(parse_args())

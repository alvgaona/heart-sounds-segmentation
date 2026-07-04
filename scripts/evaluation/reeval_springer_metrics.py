#!/usr/bin/env python3
"""Score trained CV checkpoints on Springer-comparable metrics — no retraining.

  - boundary F1: locate S1 and S2 onsets; a predicted onset is a true positive if a reference onset
    of the same sound is within +/- tolerance. Evaluated at the ORIGINAL 1000 Hz resolution
    (predictions upsampled) so it is comparable to Springer's tolerance-based numbers.
  - 2-class F1: group states into {S1, Systole} and {S2, Diastole} and compute per-frame F1 for
    each group (Springer reports F1 for "S1+systole" and "S2+diastole").

CRF/semi-CRF predictions use decode_valid (constrained-posterior, guaranteed-valid cardiac cycle); the plain
softmax LSTM (`--model lstm`, the 2020 baseline) argmaxes its per-frame emissions with no transition model. The
valid-cycle % row makes the difference explicit: the CRF is 100% by construction, the LSTM is not.
"""

import argparse

import torch
from reeval_decoders import (
    DEFAULT_LOG_DIR,
    add_split_args,
    build_test_splits,
    decode_preds,
    downsample_time,
    get_device,
    latest_ckpt,
    load_segmenter,
    valid_cycle_fraction,
)
from torchmetrics.classification import F1Score


TOLERANCES_MS = [40, 60, 100]
SOUNDS = {"S1": 0, "S2": 2}  # onset of these states


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["crf", "lstm", "tcn", "semi_crf"], required=True)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--fsst-path", default="data/springer_fsst/springer_fsst.pt")
    parser.add_argument(
        "--ref-labels-path",
        default="data/springer_fsst/springer_fsst.pt",
        help="1000 Hz labels source for onset matching (shared across configs; pre-pooled feature .pt files "
        "drop the 1000 Hz labels, so onset F1 references this full-res set instead).",
    )
    parser.add_argument("--downsample", type=int, default=20)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=68)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu", "mps"], default="cpu")
    add_split_args(parser)
    return parser.parse_args()


def onset_lists(rows: torch.Tensor, state: int) -> list[list[int]]:
    """For each row, the frame indices where `state` starts (vectorized)."""
    is_state = rows == state
    onset = is_state.clone()
    onset[:, 1:] &= ~is_state[:, :-1]
    return [torch.nonzero(onset[n]).flatten().tolist() for n in range(rows.shape[0])]


def boundary_f1(pred_onsets: list[list[int]], ref_onsets: list[list[int]], tol: int) -> float:
    """Onset-matching F1: each ref onset matches at most one predicted onset within +/- tol."""
    tp = fp = fn = 0
    for p_on, r_on in zip(pred_onsets, ref_onsets, strict=True):
        matched: set[int] = set()
        for po in p_on:
            best, best_d = None, tol + 1
            for ri, ro in enumerate(r_on):
                if ri in matched:
                    continue
                d = abs(po - ro)
                if d <= tol and d < best_d:
                    best, best_d = ri, d
            if best is not None:
                tp += 1
                matched.add(best)
            else:
                fp += 1
        fn += len(r_on) - len(matched)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def main(args: argparse.Namespace) -> None:
    log_dir = args.log_dir or DEFAULT_LOG_DIR[args.model]
    device = get_device(args.accelerator)
    factor = args.downsample
    print(f"Model: {args.model} | log-dir: {log_dir} | device: {device} | rate: {1000 // factor} Hz")

    data = torch.load(args.fsst_path, weights_only=True, mmap=factor > 1)
    features, labels_ds = downsample_time(data["features"], data["labels"], factor)
    ref_labels = torch.load(args.ref_labels_path, weights_only=True)["labels"]  # 1000 Hz onset reference
    onset_factor = ref_labels.shape[1] // features.shape[1]  # working rate -> 1000 Hz
    ref_len = onset_factor * features.shape[1]
    splits = build_test_splits(len(features), args)

    boundary: dict[tuple[str, int], list[float]] = {(s, t): [] for s in SOUNDS for t in TOLERANCES_MS}
    two_class: dict[str, list[float]] = {"S1+Systole": [], "S2+Diastole": []}
    valid_cycle: list[float] = []

    for i in range(args.folds):
        ckpt = latest_ckpt(log_dir, i)
        if ckpt is None:
            print(f"fold {i + 1}: no checkpoint, skipping")
            continue
        try:
            net = load_segmenter(args.model, ckpt, device, args.batch_size)
        except Exception as e:
            print(f"fold {i + 1}: load failed ({type(e).__name__}), skipping")
            continue
        _, _, test_idx = splits[i]
        fx = features[test_idx]
        ref_full = ref_labels[test_idx][:, :ref_len]  # 1000 Hz reference

        preds = []
        with torch.no_grad():
            for b in range(0, len(fx), args.batch_size):
                preds.append(decode_preds(net, fx[b : b + args.batch_size].to(device), args.model).cpu())
        pred_ds = torch.cat(preds)  # (n, T) at working rate
        valid_cycle.append(valid_cycle_fraction(pred_ds))
        pred_full = pred_ds.repeat_interleave(onset_factor, dim=1)  # upsample to 1000 Hz for boundary match

        # boundary F1 (1000 Hz; tolerance in ms == samples)
        for name, state in SOUNDS.items():
            p_on = onset_lists(pred_full, state)
            r_on = onset_lists(ref_full, state)
            for tol in TOLERANCES_MS:
                boundary[(name, tol)].append(boundary_f1(p_on, r_on, tol))

        # 2-class per-frame F1 at working rate (S1+Systole vs S2+Diastole)
        p2 = (pred_ds >= 2).long()
        r2 = (labels_ds[test_idx] >= 2).long()
        f1 = F1Score(task="multiclass", num_classes=2, average=None)(p2.reshape(-1), r2.reshape(-1))
        two_class["S1+Systole"].append(f1[0].item())
        two_class["S2+Diastole"].append(f1[1].item())
        print(f"fold {i + 1}: done")
        del net

    def summarize(vals: list[float]) -> str:
        v = torch.tensor(vals)
        std = v.std(unbiased=True).item() if v.numel() > 1 else 0.0
        return f"{v.mean().item():.4f} ± {std:.4f}"

    print("\n" + "=" * 66)
    print(f"{args.model} — Springer-comparable metrics (mean ± std over folds)")
    print("=" * 66)
    print("Boundary F1 (onset within tolerance, @ 1000 Hz):")
    for name in SOUNDS:
        row = "  ".join(f"±{tol}ms={summarize(boundary[(name, tol)])}" for tol in TOLERANCES_MS)
        print(f"  {name}: {row}")
    print("2-class per-frame F1:")
    for name, vals in two_class.items():
        print(f"  {name}: {summarize(vals)}")
    print("Valid-cycle fraction (S1->Systole->S2->Diastole transitions only):")
    print(f"  {summarize(valid_cycle)}  (CRF/semi-CRF are 1.0000 by construction)")


if __name__ == "__main__":
    main(parse_args())

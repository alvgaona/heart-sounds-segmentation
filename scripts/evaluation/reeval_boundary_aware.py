#!/usr/bin/env python3
"""Boundary-aware re-evaluation (Rung 1.5): how much of the S1 gap is a window-edge framing artifact?

The 2000-sample framing (stride 1000, 50%% overlap) chops cardiac cycles; S1 segments orphaned at a
window edge are missed at ~9x the interior rate (see scripts/analyze_s1_errors.py). Because windows
overlap 50%%, each orphaned cycle is interior in a neighbouring window, so the interior-only score is the
true deployable performance. This script re-scores the SAME baseline checkpoints while trimming a margin
of frames at each window edge (per-frame metrics) and excluding onsets near the edge (onset metrics),
and compares to the untrimmed (margin 0) baseline. No retraining.
"""

import argparse

import torch
from reeval_decoders import downsample_time, get_device, kfold_indices, latest_ckpt, load_segmenter
from reeval_springer_metrics import SOUNDS, TOLERANCES_MS, boundary_f1, onset_lists
from torchmetrics.classification import F1Score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["crf", "tcn", "semi_crf"], default="crf")
    parser.add_argument("--log-dir", default="lightning_logs_crf")
    parser.add_argument("--fsst-path", default="data/springer_fsst/springer_fsst.pt")
    parser.add_argument("--downsample", type=int, default=20)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=68)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu", "mps"], default="cpu")
    parser.add_argument(
        "--margins", type=int, nargs="+", default=[0, 4, 8], help="Edge frames to trim (50 Hz); 0 = baseline"
    )
    return parser.parse_args()


def keep_interior(onsets: list[list[int]], lo: int, hi: int) -> list[list[int]]:
    """Drop onsets within the edge margin [<lo or >hi] of the (upsampled) window."""
    return [[o for o in row if lo <= o <= hi] for row in onsets]


def summarize(vals: list[float]) -> str:
    v = torch.tensor(vals)
    std = v.std(unbiased=True).item() if v.numel() > 1 else 0.0
    return f"{v.mean().item():.4f} ± {std:.4f}"


def main(args: argparse.Namespace) -> None:
    device = get_device(args.accelerator)
    factor = args.downsample
    print(f"model: {args.model} | log-dir: {args.log_dir} | device: {device} | rate: {1000 // factor} Hz")

    data = torch.load(args.fsst_path, weights_only=True, mmap=factor > 1)
    labels_full = data["labels"]
    features, labels_ds = downsample_time(data["features"], labels_full, factor)
    splits = kfold_indices(len(features), args.folds, args.seed)
    t = labels_ds.shape[1]
    t2 = (labels_full.shape[1] // factor) * factor

    # per margin: lists of per-fold metrics
    macro: dict[int, list[float]] = {m: [] for m in args.margins}
    s1_f1: dict[int, list[float]] = {m: [] for m in args.margins}
    two_s1: dict[int, list[float]] = {m: [] for m in args.margins}
    onset: dict[tuple[int, str, int], list[float]] = {
        (m, s, tol): [] for m in args.margins for s in SOUNDS for tol in TOLERANCES_MS
    }

    for i in range(args.folds):
        ckpt = latest_ckpt(args.log_dir, i)
        if ckpt is None:
            print(f"fold {i + 1}: no checkpoint, skipping")
            continue
        net = load_segmenter(args.model, ckpt, device, args.batch_size)
        _, _, test_idx = splits[i]
        fx = features[test_idx]
        ref_full = labels_full[test_idx][:, :t2]
        y_ds = labels_ds[test_idx]

        preds = []
        with torch.no_grad():
            for b in range(0, len(fx), args.batch_size):
                preds.append(net.decode_valid(fx[b : b + args.batch_size].to(device)).cpu())
        pred_ds = torch.cat(preds)
        pred_full = pred_ds.repeat_interleave(factor, dim=1)
        del net

        for m in args.margins:
            lo, hi = m, t - m
            p = pred_ds[:, lo:hi].reshape(-1)
            y = y_ds[:, lo:hi].reshape(-1)
            macro[m].append(F1Score(task="multiclass", num_classes=4, average="macro")(p, y).item())
            s1_f1[m].append(F1Score(task="multiclass", num_classes=4, average=None)(p, y)[0].item())
            p2 = (pred_ds[:, lo:hi] >= 2).long().reshape(-1)
            y2 = (y_ds[:, lo:hi] >= 2).long().reshape(-1)
            two_s1[m].append(F1Score(task="multiclass", num_classes=2, average=None)(p2, y2)[0].item())

            em = m * factor
            for name, state in SOUNDS.items():
                p_on = keep_interior(onset_lists(pred_full, state), em, t2 - em)
                r_on = keep_interior(onset_lists(ref_full, state), em, t2 - em)
                for tol in TOLERANCES_MS:
                    onset[(m, name, tol)].append(boundary_f1(p_on, r_on, tol))
        print(f"fold {i + 1}: done")

    print("\n" + "=" * 74)
    print(f"{args.model} — boundary-aware re-eval (mean ± std over folds); margin in 50 Hz frames")
    print("=" * 74)
    for m in args.margins:
        tag = "baseline (full window)" if m == 0 else f"trim {m} frames (±{m * factor}ms) each edge"
        print(f"\n--- margin {m}: {tag} ---")
        print(f"  macro F1     : {summarize(macro[m])}")
        print(f"  S1 F1        : {summarize(s1_f1[m])}")
        print(f"  S1+Systole 2c: {summarize(two_s1[m])}")
        for name in SOUNDS:
            row = "  ".join(f"±{tol}ms={summarize(onset[(m, name, tol)])}" for tol in TOLERANCES_MS)
            print(f"  {name} onset F1: {row}")


if __name__ == "__main__":
    main(parse_args())

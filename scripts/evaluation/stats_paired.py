#!/usr/bin/env python3
"""Paired significance test between two trained runs on the SAME folds — no retraining.

Both runs must share seed/folds so each fold's held-out test set is identical (e.g. FSST-44 vs WSST-nv8):
that makes the per-fold metrics paired, so we can run a paired t-test + Wilcoxon signed-rank per metric.

For each fold, decodes the held-out fold with decode_valid and computes: macro F1, per-class S1 F1, and
S1/S2 onset F1 (±40 ms @ 1000 Hz). Reports mean_A, mean_B, delta (B - A), and both p-values.
"""

import argparse

import torch
from reeval_decoders import (
    downsample_time,
    get_device,
    kfold_indices,
    latest_ckpt,
    load_segmenter,
)
from reeval_springer_metrics import boundary_f1, onset_lists
from scipy import stats
from torchmetrics.classification import F1Score


METRICS = ["macro_f1", "s1_f1", "s1_onset40", "s2_onset40"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["crf", "tcn", "semi_crf"], default="crf")
    parser.add_argument(
        "--model-a", choices=["crf", "tcn", "semi_crf"], default=None, help="Run A model (default --model)"
    )
    parser.add_argument(
        "--model-b", choices=["crf", "tcn", "semi_crf"], default=None, help="Run B model (default --model)"
    )
    parser.add_argument("--log-dir-a", required=True, help="Run A (baseline) fold checkpoint root")
    parser.add_argument("--fsst-path-a", default="data/springer_fsst/springer_fsst.pt")
    parser.add_argument("--downsample-a", type=int, default=20, help="Avg-pool factor for run A features")
    parser.add_argument("--log-dir-b", required=True, help="Run B (candidate) fold checkpoint root")
    parser.add_argument("--fsst-path-b", required=True)
    parser.add_argument("--downsample-b", type=int, default=20, help="Avg-pool factor for run B features")
    parser.add_argument(
        "--ref-labels-path",
        default="data/springer_fsst/springer_fsst.pt",
        help="1000 Hz labels source for onset matching. Onset F1 must reference full-res labels; pre-pooled "
        "feature .pt files drop them, so all runs match onsets against this shared (identical) label set.",
    )
    parser.add_argument("--label-a", default="FSST")
    parser.add_argument("--label-b", default="WSST")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=68)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu", "mps"], default="cpu")
    return parser.parse_args()


def per_fold_metrics(
    model: str,
    log_dir: str,
    fsst_path: str,
    factor: int,
    ref_labels: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, list[float | None]]:
    """Per-fold metrics for one run. Missing-checkpoint folds get None (aligned across runs by index).

    Onset F1 is matched at the reference (1000 Hz) rate against `ref_labels`, independent of how the run's
    features were stored: predictions decode at the working rate then upsample by ref_len // working_len. This
    keeps the ms tolerance identical for full-res and pre-pooled configs (a pre-pooled .pt has no 1000 Hz labels).
    """
    data = torch.load(fsst_path, weights_only=True, mmap=factor > 1)
    features, labels_ds = downsample_time(data["features"], data["labels"], factor)
    splits = kfold_indices(len(features), args.folds, args.seed)
    onset_factor = ref_labels.shape[1] // features.shape[1]  # working rate -> 1000 Hz
    ref_len = onset_factor * features.shape[1]
    macro_f1 = F1Score(task="multiclass", num_classes=4, average="macro")
    per_class_f1 = F1Score(task="multiclass", num_classes=4, average=None)

    out: dict[str, list[float | None]] = {m: [] for m in METRICS}
    for i in range(args.folds):
        ckpt = latest_ckpt(log_dir, i)
        if ckpt is None:
            for m in METRICS:
                out[m].append(None)
            print(f"  fold {i + 1}: no checkpoint")
            continue
        net = load_segmenter(model, ckpt, device, args.batch_size)
        _, _, test_idx = splits[i]
        fx = features[test_idx]
        ref_full = ref_labels[test_idx][:, :ref_len]

        preds = []
        with torch.no_grad():
            for b in range(0, len(fx), args.batch_size):
                preds.append(net.decode_valid(fx[b : b + args.batch_size].to(device)).cpu())
        pred_ds = torch.cat(preds)
        pred_full = pred_ds.repeat_interleave(onset_factor, dim=1)
        y_ds = labels_ds[test_idx]

        out["macro_f1"].append(macro_f1(pred_ds.reshape(-1), y_ds.reshape(-1)).item())
        out["s1_f1"].append(per_class_f1(pred_ds.reshape(-1), y_ds.reshape(-1))[0].item())
        out["s1_onset40"].append(boundary_f1(onset_lists(pred_full, 0), onset_lists(ref_full, 0), 40))
        out["s2_onset40"].append(boundary_f1(onset_lists(pred_full, 2), onset_lists(ref_full, 2), 40))
        print(f"  fold {i + 1}: done")
        del net
    return out


def paired(a: list[float | None], b: list[float | None]) -> tuple[list[float], list[float]]:
    """Keep only folds present in both runs, preserving order."""
    pa, pb = [], []
    for xa, xb in zip(a, b, strict=True):
        if xa is not None and xb is not None:
            pa.append(xa)
            pb.append(xb)
    return pa, pb


def main(args: argparse.Namespace) -> None:
    device = get_device(args.accelerator)
    ref_labels = torch.load(args.ref_labels_path, weights_only=True)["labels"]  # 1000 Hz onset reference
    model_a = args.model_a or args.model
    model_b = args.model_b or args.model
    print(f"Onset reference labels: {args.ref_labels_path} {tuple(ref_labels.shape)}")
    print(f"Run A ({args.label_a}, {model_a}): {args.log_dir_a}  <- {args.fsst_path_a} (x{args.downsample_a})")
    ra = per_fold_metrics(model_a, args.log_dir_a, args.fsst_path_a, args.downsample_a, ref_labels, args, device)
    print(f"Run B ({args.label_b}, {model_b}): {args.log_dir_b}  <- {args.fsst_path_b} (x{args.downsample_b})")
    rb = per_fold_metrics(model_b, args.log_dir_b, args.fsst_path_b, args.downsample_b, ref_labels, args, device)

    print("\n" + "=" * 78)
    print(f"Paired {args.label_b} - {args.label_a}  (delta > 0 favors {args.label_b}; n paired folds)")
    print("=" * 78)
    header = f"{'metric':<12} {args.label_a:>9} {args.label_b:>9} {'delta':>9} {'std':>8} {'t p':>8} {'wilcox p':>9}  n"
    print(header)
    for m in METRICS:
        a, b = paired(ra[m], rb[m])
        if len(a) < 2:
            print(f"{m:<12} insufficient paired folds ({len(a)})")
            continue
        ta, tb = torch.tensor(a), torch.tensor(b)
        delta = tb - ta
        t_p = float(stats.ttest_rel(tb.numpy(), ta.numpy()).pvalue)
        try:
            w_p = float(stats.wilcoxon(tb.numpy(), ta.numpy()).pvalue)
        except ValueError:
            w_p = float("nan")
        print(
            f"{m:<12} {ta.mean():>9.4f} {tb.mean():>9.4f} {delta.mean():>+9.4f} "
            f"{delta.std(unbiased=True):>8.4f} {t_p:>8.4f} {w_p:>9.4f}  {len(a)}"
        )


if __name__ == "__main__":
    main(parse_args())

"""Data-efficiency curve: re-score sweep checkpoints (arch x train-fraction) under decode_valid.

Trains are produced by ``train_crf.py --train-fraction f --arch {bilstm,xlstm} --split-by patient``,
one log-dir per (arch, fraction). This scores each on the SAME full held-out test folds (the test fold
is unaffected by --train-fraction, so fractions and archs are paired) and prints the macro-F1 curve.

    pixi run python scripts/evaluation/data_efficiency_curve.py \
        --archs bilstm xlstm --fractions 0.05 0.1 0.25 0.5 1.0 --folds 3
"""

import argparse
import os
from types import SimpleNamespace

import torch
from reeval_decoders import build_test_splits, downsample_time, get_device, latest_ckpt, load_segmenter
from torchmetrics import F1Score


def pct_tag(frac: float) -> str:
    return f"{int(round(frac * 100)):02d}"  # 0.05->05, 0.1->10, 1.0->100


def score_logdir(log_dir: str, arch: str, args: SimpleNamespace, feats, labs, device) -> list[float]:
    """decode_valid macro F1 per fold for one sweep checkpoint dir (empty if the dir is missing)."""
    if not os.path.isdir(log_dir):
        return []
    splits = build_test_splits(len(feats), args)
    f1s = []
    for i in range(args.folds):
        ckpt = latest_ckpt(log_dir, i)
        if ckpt is None:
            continue
        net = load_segmenter("crf", ckpt, device, args.batch_size, arch)
        _, _, test_idx = splits[i]
        fx, yx = feats[test_idx], labs[test_idx]
        preds = []
        with torch.no_grad():
            for b in range(0, len(fx), args.batch_size):
                preds.append(net.decode_valid(fx[b : b + args.batch_size].to(device)).cpu())
        pred = torch.cat(preds).reshape(-1)
        f1s.append(F1Score(task="multiclass", num_classes=4, average="macro")(pred, yx.reshape(-1)).item())
        del net
    return f1s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", nargs="+", default=["bilstm", "xlstm"])
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--template", default="lightning_logs_crf_{arch}_f{pct}_fsst_patient")
    ap.add_argument("--fsst-path", default="data/springer_fsst/springer_fsst.pt")
    ap.add_argument("--downsample", type=int, default=20)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=68)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--accelerator", choices=["auto", "cpu", "gpu", "mps"], default="cpu")
    ap.add_argument("--split-by", choices=["frame", "recording", "patient"], default="patient")
    ap.add_argument("--patient-index", default="data/patient_ids.pt")
    ap.add_argument("--recording-index", default="data/recording_ids.pt")
    args = ap.parse_args()

    device = get_device(args.accelerator)
    data = torch.load(args.fsst_path, weights_only=True, mmap=args.downsample > 1)
    feats, labs = downsample_time(data["features"], data["labels"], args.downsample)

    cell: dict[tuple[str, float], str] = {}
    for arch in args.archs:
        for frac in args.fractions:
            log_dir = args.template.format(arch=arch, pct=pct_tag(frac))
            f1s = score_logdir(log_dir, arch, args, feats, labs, device)
            t = torch.tensor(f1s)
            cell[(arch, frac)] = f"{t.mean():.4f}±{t.std():.3f}" if f1s else "   —   "
            print(f"scored {arch:6s} f={frac:<4} folds={len(f1s)}  {cell[(arch, frac)]}")

    print("\n" + "=" * (14 + 16 * len(args.archs)))
    print("Data-efficiency curve — decode_valid macro F1 (mean±std over folds)")
    print("=" * (14 + 16 * len(args.archs)))
    header = f"{'% labels':<12}" + "".join(f"{a:>16}" for a in args.archs)
    print(header)
    print("-" * len(header))
    for frac in args.fractions:
        row = f"{int(round(frac * 100)):>6}%     " + "".join(f"{cell[(a, frac)]:>16}" for a in args.archs)
        print(row)


if __name__ == "__main__":
    main()

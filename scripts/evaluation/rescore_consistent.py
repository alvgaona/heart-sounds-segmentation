#!/usr/bin/env python3
"""Per-fold macro + per-class F1 for all three models under one decoder (decode_valid), plus paired
significance tests (paired t-test + Wilcoxon signed-rank). One consistent basis for the RESULTS.md tables,
so BiTCN's clean lr=0.01 numbers are not spliced against a different decoder or the older lr=1e-3 run.

Requires local checkpoints in lightning_logs_{crf,tcn,semi_crf}/ and data/springer_fsst/springer_fsst.pt.
Runs on CPU (the semi-CRF has an MPS log_Z bug). Reuses the helpers in reeval_decoders.py."""

import torch
from reeval_decoders import (
    downsample_time,
    get_device,
    kfold_indices,
    latest_ckpt,
    load_segmenter,
)
from scipy import stats
from torchmetrics.classification import F1Score


MODELS = {"crf": "lightning_logs_crf", "tcn": "lightning_logs_tcn", "semi_crf": "lightning_logs_semi_crf"}
FSST = "data/springer_fsst/springer_fsst.pt"
FACTOR = 20
FOLDS = 10
SEED = 68
BS = 50


def score_model(name: str, log_dir: str, features: torch.Tensor, labels: torch.Tensor, splits) -> dict:
    device = get_device("cpu")
    macro: list[float] = []
    per_class: list[list[float]] = []
    for i in range(FOLDS):
        ckpt = latest_ckpt(log_dir, i)
        net = load_segmenter(name, ckpt, device, BS)
        _, _, test_idx = splits[i]
        fx, yx = features[test_idx], labels[test_idx]
        preds = []
        with torch.no_grad():
            for b in range(0, len(fx), BS):
                preds.append(net.decode_valid(fx[b : b + BS].to(device)).cpu())
        p = torch.cat(preds).reshape(-1)
        y = yx.reshape(-1)
        macro.append(F1Score(task="multiclass", num_classes=4, average="macro")(p, y).item())
        pc = F1Score(task="multiclass", num_classes=4, average=None)(p, y)
        per_class.append([pc[c].item() for c in range(4)])
        del net
    return {"macro": macro, "per_class": per_class}


def main() -> None:
    data = torch.load(FSST, weights_only=True, mmap=True)
    features, labels = downsample_time(data["features"], data["labels"], FACTOR)
    splits = kfold_indices(len(features), FOLDS, SEED)

    res = {}
    for name, log_dir in MODELS.items():
        res[name] = score_model(name, log_dir, features, labels, splits)
        m = torch.tensor(res[name]["macro"])
        print(f"\n=== {name} (decode_valid) ===")
        print("per-fold macro:", "  ".join(f"{v:.4f}" for v in res[name]["macro"]))
        print(f"macro F1: {m.mean():.4f} +/- {m.std(unbiased=True):.4f}")
        pc = torch.tensor(res[name]["per_class"]).mean(0)
        classes = ["S1", "Systole", "S2", "Diastole"]
        print("per-class:", "  ".join(f"{c}={pc[j]:.4f}" for j, c in enumerate(classes)))

    def paired(a_name: str, b_name: str) -> None:
        a = torch.tensor(res[a_name]["macro"])
        b = torch.tensor(res[b_name]["macro"])
        d = a - b
        wins = int((d > 0).sum())
        t_p = stats.ttest_rel(a.numpy(), b.numpy()).pvalue
        w_p = stats.wilcoxon(a.numpy(), b.numpy()).pvalue
        print(
            f"{a_name:8s} - {b_name:8s}: mean d={d.mean():+.4f}  wins={wins}/{FOLDS}  "
            f"t_p={t_p:.4f}  wilcoxon_p={w_p:.4f}"
        )

    print("\n=== paired tests (macro F1, decode_valid) ===")
    paired("semi_crf", "crf")
    paired("semi_crf", "tcn")
    paired("crf", "tcn")


if __name__ == "__main__":
    main()

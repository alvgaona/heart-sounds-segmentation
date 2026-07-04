#!/usr/bin/env python3
"""Cross-dataset generalization: decode CirCor with Springer-trained checkpoints (C7). No retraining.

Each Springer fold checkpoint (trained + tested only on Springer) decodes ALL CirCor frames; metrics are
averaged over the fold checkpoints (mean +/- std = generalization estimate with fold variance). CirCor has
unannotated `-1` frames, so every metric EXCLUDES `-1`:
  - per-frame macro F1 and 2-class F1: computed only where label != -1.
  - onset F1 (S1/S2, +/-40/60/100 ms @ 1000 Hz): predicted onsets landing in a `-1` reference span are dropped;
    references (labels_hr) are naturally in annotated regions. Pooled (micro over frames) AND per-patient mean.
  - valid-cycle fraction: per 2 s window (windowed protocol).

Reference onsets come from CirCor's own `labels_hr` (its hybrid HSMM+expert segmentation) — a partly circular
reference vs our HSMM-lineage model; documented as a threat to validity. Feed `--circor-path` a
precompute_circor.py output (features already pooled to the model's rate; `labels_hr` at 1000 Hz).
"""

import argparse
from collections import defaultdict

import torch
from reeval_decoders import (
    DEFAULT_LOG_DIR,
    decode_preds,
    get_device,
    latest_ckpt,
    load_segmenter,
    valid_cycle_fraction,
)
from reeval_springer_metrics import boundary_f1, onset_lists
from torchmetrics.classification import F1Score


TOLERANCES_MS = [40, 60, 100]
SOUNDS = {"S1": 0, "S2": 2}  # 0-indexed states whose onsets we score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["crf", "lstm", "semi_crf"], default="crf")
    parser.add_argument("--log-dir", default=None, help="Springer fold-checkpoint root (default per model)")
    parser.add_argument("--circor-path", default="data/circor_wsst_nv8/circor_wsst.pt")
    parser.add_argument("--folds", type=int, default=10, help="Number of Springer fold checkpoints to average")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu", "mps"], default="cpu")
    return parser.parse_args()


def drop_masked_onsets(
    onsets_per_row: list[list[int]], neg_mask: torch.Tensor, has_neg: torch.Tensor
) -> list[list[int]]:
    """Drop predicted onsets that fall in an unannotated (-1) reference span (skip rows with no -1)."""
    return [row if not has_neg[n] else [o for o in row if not neg_mask[n, o]] for n, row in enumerate(onsets_per_row)]


def onset_f1_grouped(p_on: list[list[int]], r_on: list[list[int]], group_idx: list[list[int]], tol: int) -> float:
    """Mean over groups (precomputed row-index lists) of per-group boundary F1."""
    vals = [boundary_f1([p_on[i] for i in idx], [r_on[i] for i in idx], tol) for idx in group_idx]
    return float(torch.tensor(vals).mean())


def main(args: argparse.Namespace) -> None:
    log_dir = args.log_dir or DEFAULT_LOG_DIR[args.model]
    device = get_device(args.accelerator)
    data = torch.load(args.circor_path, weights_only=False)
    feats, labs = data["features"], data["labels"]
    labs_hr = data["labels_hr"].long()
    pat = data["patient_id"]
    onset_factor = labs_hr.shape[1] // feats.shape[1]
    print(f"Model {args.model} | Springer ckpts {log_dir} -> CirCor {args.circor_path}")
    print(f"CirCor: {len(feats)} frames, {len(torch.unique(pat))} patients, rate {feats.shape[1] // 2} Hz (2 s window)")

    acc: dict[str, list[float]] = defaultdict(list)

    # Fold-invariant precompute: reference onsets, -1 mask, and per-patient row indices.
    neg_mask = labs_hr == -1
    has_neg = neg_mask.any(dim=1)
    ref_onsets = {state: onset_lists(labs_hr, state) for state in SOUNDS.values()}
    group_idx = [(pat == g).nonzero().flatten().tolist() for g in torch.unique(pat).tolist()]

    for i in range(args.folds):
        ckpt = latest_ckpt(log_dir, i)
        if ckpt is None:
            print(f"fold {i + 1}: no checkpoint, skipping")
            continue
        net = load_segmenter(args.model, ckpt, device, args.batch_size)
        preds = []
        with torch.no_grad():
            for b in range(0, len(feats), args.batch_size):
                preds.append(decode_preds(net, feats[b : b + args.batch_size].to(device), args.model).cpu())
        pred = torch.cat(preds)  # (N, T) at working rate
        pred_hr = pred.repeat_interleave(onset_factor, dim=1)

        mask = labs != -1  # per-frame annotated mask
        p_flat, y_flat = pred[mask], labs[mask]
        acc["macro_f1"].append(F1Score(task="multiclass", num_classes=4, average="macro")(p_flat, y_flat).item())
        f2 = F1Score(task="multiclass", num_classes=2, average=None)((p_flat >= 2).long(), (y_flat >= 2).long())
        acc["s1_sys_f1"].append(f2[0].item())
        acc["s2_dia_f1"].append(f2[1].item())
        acc["valid_cycle"].append(valid_cycle_fraction(pred))

        for name, state in SOUNDS.items():
            p_on = drop_masked_onsets(onset_lists(pred_hr, state), neg_mask, has_neg)
            r_on = ref_onsets[state]
            for tol in TOLERANCES_MS:
                acc[f"{name}_on{tol}"].append(boundary_f1(p_on, r_on, tol))
                acc[f"{name}_on{tol}_pp"].append(onset_f1_grouped(p_on, r_on, group_idx, tol))
        print(f"fold {i + 1}: done")
        del net

    def summ(v: list[float]) -> str:
        t = torch.tensor(v)
        return f"{t.mean():.4f} +/- {t.std(unbiased=True):.4f}" if t.numel() > 1 else f"{t.mean():.4f}"

    print("\n" + "=" * 72)
    print(f"Springer -> CirCor generalization ({args.model}, mean +/- std over {args.folds} Springer checkpoints)")
    print("=" * 72)
    print("Per-frame (masked, exclude -1):")
    print(f"  macro F1     {summ(acc['macro_f1'])}")
    print(f"  2-class F1   S1+Sys {summ(acc['s1_sys_f1'])} | S2+Dia {summ(acc['s2_dia_f1'])}")
    print(f"  valid-cycle  {summ(acc['valid_cycle'])} (per 2 s window)")
    print("Onset F1 @ 1000 Hz (pooled | per-patient):")
    for s in SOUNDS:
        for t in TOLERANCES_MS:
            print(f"  {s} +/-{t}ms  {summ(acc[f'{s}_on{t}'])}  |  pp {summ(acc[f'{s}_on{t}_pp'])}")


if __name__ == "__main__":
    main(parse_args())

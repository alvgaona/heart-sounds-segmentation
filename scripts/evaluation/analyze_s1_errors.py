#!/usr/bin/env python3
"""Pass 1 error analysis: why does the LSTM+CRF baseline miss S1? No retraining.

Loads each fold's baseline checkpoint, decodes the held-out fold with decode_valid, and reports:
  - S1 frame-level confusion (what true-S1 frames get predicted as) + spurious-S1 composition
  - whole-segment miss rate (GT S1 segments predicted <50% S1) and what they get called instead
  - per-recording S1 recall histogram (are misses concentrated in a few records, or spread everywhere?)
  - a PNG gallery of missed-S1 windows: the FSST heatmap the model sees + GT + prediction + P(S1)

The gallery answers the key question: at a missed S1, is S1 even visible in the FSST the model consumes?
  - flat FSST at the miss  -> feature gap  -> envelope fusion (Rung 2)
  - clear burst, missed     -> model gap    -> capacity / loss / architecture
"""

import argparse
import os

import matplotlib
import torch


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reeval_decoders import downsample_time, get_device, kfold_indices, latest_ckpt, load_segmenter  # noqa: E402


CLASS_NAMES = ["S1", "Systole", "S2", "Diastole"]
S1 = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="lightning_logs_crf")
    parser.add_argument("--fsst-path", default="data/springer_fsst/springer_fsst.pt")
    parser.add_argument("--downsample", type=int, default=20)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=68)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu", "mps"], default="cpu")
    parser.add_argument("--out-dir", default="s1_error_gallery", help="Where to write the PNG gallery")
    parser.add_argument("--gallery", type=int, default=24, help="Max missed-S1 windows to render")
    parser.add_argument("--miss-thresh", type=float, default=0.5, help="Segment recall below this = missed")
    parser.add_argument("--window", type=int, default=25, help="Half-width (frames) of each gallery window")
    return parser.parse_args()


def runs_of(row: torch.Tensor, cls: int) -> list[tuple[int, int]]:
    """Contiguous [start, end) index ranges where row == cls."""
    mask = (row == cls).tolist()
    out: list[tuple[int, int]] = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def plot_event(event: dict, path: str) -> None:
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1.5]})
    ax0.imshow(event["feat"].T.numpy(), aspect="auto", origin="lower", cmap="viridis")
    ax0.set_ylabel("FSST feat")
    ax0.set_title(event["title"])

    strip = torch.stack([event["gt"], event["pred"]]).numpy()
    ax1.imshow(strip, aspect="auto", cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["GT", "pred"])

    xs = range(event["gt"].shape[0])
    ax2.plot(event["ps1"].numpy(), color="tab:blue")
    ax2.axhline(0.5, ls="--", lw=0.5, color="gray")
    ax2.fill_between(xs, 0, 1, where=(event["gt"] == S1).numpy(), alpha=0.2, color="red", label="GT S1")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("P(S1)")
    ax2.set_xlabel("frame (50 Hz)")
    ax2.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    device = get_device(args.accelerator)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"log-dir: {args.log_dir} | device: {device} | rate: {1000 // args.downsample} Hz")

    data = torch.load(args.fsst_path, weights_only=True, mmap=args.downsample > 1)
    features, labels = downsample_time(data["features"], data["labels"], args.downsample)
    splits = kfold_indices(len(features), args.folds, args.seed)

    confusion = [0, 0, 0, 0]  # true-S1 frames -> predicted class
    total_s1 = 0
    spurious = [0, 0, 0, 0]  # pred-S1 frames -> true class
    total_pred_s1 = 0
    seg_recalls: list[float] = []
    missed_pred_class = [0, 0, 0, 0]  # missed S1 segments -> majority predicted class
    rec_hist_bins = [0.0, 0.5, 0.7, 0.9, 1.0001]
    rec_hist = [0, 0, 0, 0]
    seg_edge = [0, 0]  # [interior, boundary-touching] total S1 segments
    seg_edge_missed = [0, 0]  # [interior, boundary-touching] missed S1 segments
    per_fold_cap = max(1, args.gallery // args.folds)
    gallery: list[dict] = []

    for i in range(args.folds):
        ckpt = latest_ckpt(args.log_dir, i)
        if ckpt is None:
            print(f"fold {i + 1}: no checkpoint, skipping")
            continue
        net = load_segmenter("crf", ckpt, device, args.batch_size)
        _, _, test_idx = splits[i]
        fx, fy = features[test_idx], labels[test_idx]
        preds, posts = [], []
        with torch.no_grad():
            for b in range(0, len(fx), args.batch_size):
                xb = fx[b : b + args.batch_size].to(device)
                preds.append(net.decode_valid(xb).cpu())
                posts.append(net.marginals(xb).cpu())
        pred = torch.cat(preds)
        post = torch.cat(posts)
        del net

        s1_frames = fy == S1
        total_s1 += int(s1_frames.sum())
        pred_s1_frames = pred == S1
        total_pred_s1 += int(pred_s1_frames.sum())
        for c in range(4):
            confusion[c] += int((pred == c)[s1_frames].sum())
            spurious[c] += int((fy == c)[pred_s1_frames].sum())

        fold_gallery = 0
        for n in range(len(fx)):
            yr, pr = fy[n], pred[n]
            tot = int((yr == S1).sum())
            if tot > 0:
                rec = int(((pr == S1) & (yr == S1)).sum()) / tot
                for bidx in range(4):
                    if rec_hist_bins[bidx] <= rec < rec_hist_bins[bidx + 1]:
                        rec_hist[bidx] += 1
                        break
            for s, e in runs_of(yr, S1):
                seg_recall = float((pr[s:e] == S1).float().mean())
                seg_recalls.append(seg_recall)
                edge = 1 if (s == 0 or e == fy.shape[1]) else 0
                seg_edge[edge] += 1
                if seg_recall < args.miss_thresh:
                    seg_edge_missed[edge] += 1
                    maj = int(pr[s:e].mode().values)
                    missed_pred_class[maj] += 1
                    if fold_gallery < per_fold_cap:
                        c = (s + e) // 2
                        w0, w1 = max(0, c - args.window), min(fy.shape[1], c + args.window)
                        gallery.append(
                            {
                                "feat": fx[n, w0:w1].clone(),
                                "gt": yr[w0:w1].clone(),
                                "pred": pr[w0:w1].clone(),
                                "ps1": post[n, w0:w1, S1].clone(),
                                "title": f"fold{i + 1} rec{n} seg[{s}:{e}] recall={seg_recall:.2f} "
                                f"-> {CLASS_NAMES[maj]}",
                            }
                        )
                        fold_gallery += 1
        print(f"fold {i + 1}: done ({len(fx)} records)")

    _report(confusion, total_s1, spurious, total_pred_s1, seg_recalls, missed_pred_class, rec_hist, args)

    print("\nBoundary artifact check (S1 segments touching frame 0 or the last frame):")
    for lab, idx in (("interior", 0), ("boundary", 1)):
        tot = max(seg_edge[idx], 1)
        print(f"  {lab:8s}: {seg_edge_missed[idx]:4d}/{seg_edge[idx]:5d} missed ({seg_edge_missed[idx] / tot:.2%})")
    tot_missed = max(seg_edge_missed[0] + seg_edge_missed[1], 1)
    print(f"  boundary share of ALL misses: {seg_edge_missed[1] / tot_missed:.2%}")

    for k, event in enumerate(gallery):
        plot_event(event, os.path.join(args.out_dir, f"missed_s1_{k:02d}.png"))
    print(f"\nGallery: {len(gallery)} missed-S1 windows -> {args.out_dir}/missed_s1_*.png")


def _report(
    confusion: list[int],
    total_s1: int,
    spurious: list[int],
    total_pred_s1: int,
    seg_recalls: list[float],
    missed_pred_class: list[int],
    rec_hist: list[int],
    args: argparse.Namespace,
) -> None:
    print("\n" + "=" * 66)
    print("S1 ERROR ANALYSIS (baseline LSTM+CRF, decode_valid)")
    print("=" * 66)

    print(f"\nTrue-S1 frames: {total_s1}  (per-frame S1 recall = {confusion[S1] / max(total_s1, 1):.4f})")
    print("Where true-S1 frames go (confusion):")
    for c in range(4):
        print(f"  -> {CLASS_NAMES[c]:9s}: {confusion[c] / max(total_s1, 1):6.2%}")

    print(f"\nPred-S1 frames: {total_pred_s1}  (S1 precision = {spurious[S1] / max(total_pred_s1, 1):.4f})")
    print("What pred-S1 frames really are (spurious composition):")
    for c in range(4):
        print(f"  is {CLASS_NAMES[c]:9s}: {spurious[c] / max(total_pred_s1, 1):6.2%}")

    n_seg = len(seg_recalls)
    missed = sum(1 for r in seg_recalls if r < args.miss_thresh)
    fully = sum(1 for r in seg_recalls if r == 0.0)
    print(f"\nGT S1 segments: {n_seg}")
    print(f"  missed (recall < {args.miss_thresh}): {missed} ({missed / max(n_seg, 1):.2%})")
    print(f"  fully missed (recall == 0):       {fully} ({fully / max(n_seg, 1):.2%})")
    tot_missed = max(sum(missed_pred_class), 1)
    print("  missed S1 segments get called:")
    for c in range(4):
        print(f"    {CLASS_NAMES[c]:9s}: {missed_pred_class[c] / tot_missed:6.2%}")

    labels_bin = ["[0.0,0.5)", "[0.5,0.7)", "[0.7,0.9)", "[0.9,1.0]"]
    tot_rec = max(sum(rec_hist), 1)
    print("\nPer-recording S1 recall histogram (concentration check):")
    for lab, cnt in zip(labels_bin, rec_hist, strict=True):
        print(f"  {lab}: {cnt:4d} ({cnt / tot_rec:.2%})")


if __name__ == "__main__":
    main(parse_args())

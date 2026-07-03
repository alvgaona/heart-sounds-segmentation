#!/usr/bin/env python3
"""Pass 2: at interior faint-S1 misses, is S1 visible in the raw signal / amplitude envelope even
though it is faint in FSST?

Loads the raw PCG frames (re-framed from the dataset, aligned 1:1 to the precomputed FSST .pt), finds
the interior missed-S1 windows of the baseline LSTM+CRF, and overlays the homomorphic + Hilbert
envelopes on the raw waveform next to the (flat) FSST. Verdict:
  - envelope shows an S1 bump the FSST misses -> feature fusion (Rung 2) will help
  - envelope is also flat                     -> near-silent S1, label/signal ceiling

The .pt has no recording IDs, but the dataset frames recordings in deterministic sorted order with the
same frame_signal(x, y-1, stride=1000, len=2000), so raw frame k == FSST frame k (asserted via labels).
"""

import argparse
import os

import matplotlib
import numpy as np
import torch


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from analyze_s1_errors import runs_of  # noqa: E402
from reeval_decoders import downsample_time, get_device, kfold_indices, latest_ckpt, load_segmenter  # noqa: E402
from scipy.signal import butter, filtfilt, hilbert  # noqa: E402

from hss.datasets import DavidSpringerHSS  # noqa: E402


S1 = 0
FS = 1000
FACTOR = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="lightning_logs_crf")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--fsst-path", default="data/springer_fsst/springer_fsst.pt")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--decode-folds", type=int, default=3, help="Decode only the first N folds (enough misses)")
    parser.add_argument("--seed", type=int, default=68)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu", "mps"], default="cpu")
    parser.add_argument("--out-dir", default="s1_pass2_gallery")
    parser.add_argument("--gallery", type=int, default=12)
    parser.add_argument("--miss-thresh", type=float, default=0.5)
    return parser.parse_args()


def bandpass(x: np.ndarray, lo: int = 25, hi: int = 400) -> np.ndarray:
    b, a = butter(2, [2 * lo / FS, 2 * hi / FS], "band")
    return filtfilt(b, a, x)


def homomorphic_envelope(x: np.ndarray, cutoff: int = 8) -> np.ndarray:
    b, a = butter(1, 2 * cutoff / FS, "low")
    return np.exp(filtfilt(b, a, np.log(np.abs(hilbert(x)) + 1e-8)))


def unit(v: np.ndarray) -> np.ndarray:
    v = v - v.min()
    m = v.max()
    return v / m if m > 0 else v


def plot_event(ev: dict, path: str) -> None:
    t_raw = np.arange(ev["raw"].shape[0]) / FS
    t_ds = np.arange(ev["heat"].shape[0]) / (FS / FACTOR)
    gt_s1_ds = ev["gt"] == S1

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1.5]})
    ax0.plot(t_raw, unit(ev["raw"]) * 2 - 1, color="0.6", lw=0.6, label="raw PCG (bandpassed)")
    ax0.plot(t_raw, unit(ev["homo"]), color="tab:blue", lw=1.6, label="homomorphic env")
    ax0.plot(t_raw, unit(ev["hilb"]), color="tab:orange", lw=1.0, label="Hilbert env")
    ax0.fill_between(t_ds, -1, 1, where=gt_s1_ds, alpha=0.2, color="red", label="GT S1")
    ax0.set_title(ev["title"])
    ax0.legend(loc="upper right", fontsize=7, ncol=2)

    ax1.imshow(ev["heat"].T, aspect="auto", origin="lower", cmap="viridis", extent=(0.0, float(t_raw[-1]), 0.0, 44.0))
    ax1.set_ylabel("FSST feat")

    ax2.plot(t_ds, ev["ps1"], color="tab:blue")
    ax2.axhline(0.5, ls="--", lw=0.5, color="gray")
    ax2.fill_between(t_ds, 0, 1, where=gt_s1_ds, alpha=0.2, color="red")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("P(S1)")
    ax2.set_xlabel("time (s)")

    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    device = get_device(args.accelerator)
    os.makedirs(args.out_dir, exist_ok=True)

    data = torch.load(args.fsst_path, weights_only=True, mmap=True)
    feats_ds, labels_ds = downsample_time(data["features"], data["labels"], FACTOR)
    n_total, t_ds = labels_ds.shape

    print("Loading raw waveforms (re-framing the dataset)...")
    raw_ds = DavidSpringerHSS(args.data_root, download=False, in_memory=True, framing=True, transform=None)
    assert len(raw_ds.data) == n_total, f"raw {len(raw_ds.data)} != fsst {n_total} (alignment broken)"
    # sanity: labels must match 1:1 (both are framed y-1)
    for k in (0, n_total // 2, n_total - 1):
        assert torch.equal(raw_ds.data[k][1], data["labels"][k]), f"label mismatch at {k}"
    print(f"Aligned {n_total} raw frames to FSST frames (labels match).")

    splits = kfold_indices(n_total, args.folds, args.seed)
    gallery: list[dict] = []

    for i in range(args.decode_folds):
        if len(gallery) >= args.gallery:
            break
        net = load_segmenter("crf", latest_ckpt(args.log_dir, i), device, args.batch_size)
        _, _, test_idx = splits[i]
        fx = feats_ds[test_idx]
        preds, posts = [], []
        with torch.no_grad():
            for b in range(0, len(fx), args.batch_size):
                xb = fx[b : b + args.batch_size].to(device)
                preds.append(net.decode_valid(xb).cpu())
                posts.append(net.marginals(xb).cpu())
        pred, post = torch.cat(preds), torch.cat(posts)
        del net

        for n in range(len(fx)):
            if len(gallery) >= args.gallery:
                break
            yr, pr = labels_ds[test_idx][n], pred[n]
            for s, e in runs_of(yr, S1):
                if s == 0 or e == t_ds:  # interior only
                    continue
                if float((pr[s:e] == S1).float().mean()) >= args.miss_thresh:
                    continue
                gidx = test_idx[n]
                c = (s + e) // 2
                w0, w1 = max(0, c - 30), min(t_ds, c + 30)
                raw = bandpass(raw_ds.data[gidx][0].squeeze().numpy().astype(np.float64))
                gallery.append(
                    {
                        "raw": raw[w0 * FACTOR : w1 * FACTOR],
                        "homo": homomorphic_envelope(raw)[w0 * FACTOR : w1 * FACTOR],
                        "hilb": np.abs(hilbert(raw))[w0 * FACTOR : w1 * FACTOR],
                        "heat": feats_ds[gidx, w0:w1].numpy(),
                        "gt": yr[w0:w1],
                        "ps1": post[n, w0:w1, S1].numpy(),
                        "title": f"fold{i + 1} rec{n} (global {gidx}) seg[{s}:{e}] "
                        f"recall={float((pr[s:e] == S1).float().mean()):.2f}",
                    }
                )
                break
        print(f"fold {i + 1}: collected {len(gallery)} interior misses")

    for k, ev in enumerate(gallery):
        plot_event(ev, os.path.join(args.out_dir, f"pass2_{k:02d}.png"))
    print(f"\nPass 2: {len(gallery)} envelope overlays -> {args.out_dir}/pass2_*.png")


if __name__ == "__main__":
    main(parse_args())

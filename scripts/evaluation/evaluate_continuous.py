#!/usr/bin/env python3
"""Continuous (full-recording) inference vs windowed decode for a trained segmenter — no retraining.

Isolates the frame-boundary artifact by decoding the SAME model and SAME recordings three ways:
  - windowed : each T=100 (2 s @ 50 Hz) window decoded independently, every frame scored (the current framed
               protocol; includes window-edge frames that lack context — the artifact).
  - stitch   : overlapping windows decoded, each timepoint taken from the window where it is most central
               (overlap-stitch; removes the edge artifact, no length extrapolation — the deployable fix).
  - continuous: the whole recording decoded in one pass (Springer-style; no artificial edges, but the model
               was trained on T=100 chunks so this extrapolates in length).

Stage 1: additive, touches no existing code. It runs the EXISTING frame-level checkpoints, so absolute levels
are leaked-optimistic (the frame-level CV split scatters each recording across all folds). Leakage is identical
across the three strategies, so the windowed->stitch/continuous DELTA cleanly estimates the framing-artifact
cost. A leakage-free number needs recording-level retraining (Stage 2).

Caveat: WSST here is computed per full recording (z-norm over the whole recording) vs per-2s-frame at train
time; heart sounds are quasi-stationary so this should be minor, but it is the first suspect if continuous
underperforms.
"""

import argparse

import torch
from reeval_decoders import get_device, latest_ckpt, load_segmenter
from reeval_springer_metrics import boundary_f1, onset_lists
from torchmetrics.functional import f1_score

from hss.datasets import DavidSpringerHSS
from hss.transforms import WSST


TOLERANCES_MS = [40, 60, 100]
SOUNDS = {"S1": 0, "S2": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["crf", "tcn", "semi_crf"], default="crf")
    parser.add_argument("--log-dir", default="lightning_logs_crf_wsst_nv8")
    parser.add_argument("--dataset-path", default="data")
    parser.add_argument("--wavelet", choices=["amor", "bump"], default="amor")
    parser.add_argument("--num-voices", type=int, default=8)
    parser.add_argument("--truncate-freq", type=float, nargs=2, default=(25.0, 200.0), metavar=("FMIN", "FMAX"))
    parser.add_argument("--downsample", type=int, default=20, help="Feature/label pooling factor (1000 Hz -> 50 Hz)")
    parser.add_argument("--frame-len", type=int, default=2000, help="Raw window length (samples) matching training")
    parser.add_argument("--stride", type=int, default=1000, help="Raw window stride (samples) matching training")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-recordings", type=int, default=None, help="Cap recordings (smoke test)")
    parser.add_argument(
        "--split-by",
        choices=["all", "recording", "patient"],
        default="all",
        help="'all' (Stage 1): every recording under every fold model (leaked). 'recording'/'patient': only "
        "each fold's held-out recordings/patients (leakage-free; use with the matching checkpoints).",
    )
    parser.add_argument("--recording-index", default="data/recording_ids.pt")
    parser.add_argument("--patient-index", default="data/patient_ids.pt")
    parser.add_argument("--seed", type=int, default=68, help="Split seed (must match training)")
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu", "mps"], default="cpu")
    return parser.parse_args()


def downsample_time(features: torch.Tensor, labels: torch.Tensor, factor: int) -> tuple[torch.Tensor, torch.Tensor]:
    if factor <= 1:
        return features, labels
    t2 = (features.shape[0] // factor) * factor
    f = features[:t2].reshape(t2 // factor, factor, -1).mean(dim=1)
    lab = labels[:t2].reshape(t2 // factor, factor).mode(dim=1).values
    return f, lab


def window_starts(length: int, win: int, stride: int) -> list[int]:
    """Window starts covering [0, length); the last window snaps to end exactly at `length`."""
    if length <= win:
        return [0]
    starts = list(range(0, length - win + 1, stride))
    if starts[-1] != length - win:
        starts.append(length - win)
    return starts


def stitch_central(preds_win: torch.Tensor, starts: list[int], win: int, length: int) -> torch.Tensor:
    """Overlap-stitch: each timepoint takes the prediction from the window where it is most central."""
    out = torch.zeros(length, dtype=torch.long)
    best = torch.full((length,), win + 1, dtype=torch.long)
    for w, s in enumerate(starts):
        end = min(s + win, length)
        idx = torch.arange(s, end)
        d = (idx - (s + win // 2)).abs()
        take = d < best[idx]
        sel = idx[take]
        out[sel] = preds_win[w, (sel - s)]
        best[sel] = d[take]
    return out


def f1(preds: torch.Tensor, target: torch.Tensor, average: str) -> torch.Tensor:
    return f1_score(preds, target, task="multiclass", num_classes=4, average=average)


def recording_test_folds(n_rec: int, k: int, seed: int) -> list[list[int]]:
    """Held-out recording indices per fold (must match training's recording-level split)."""
    perm = torch.randperm(n_rec, generator=torch.Generator().manual_seed(seed)).tolist()
    return [perm[i::k] for i in range(k)]


def patient_test_folds(recording_index_path: str, patient_index_path: str, k: int, seed: int) -> list[list[int]]:
    """Held-out RECORDING indices per fold when splitting by PATIENT (matches training's patient-level split).

    Splits the patients (same sorted-unique + seeded perm as grouped_kfold), then returns, per fold, the
    recordings whose patient is held out. Recording index r is the r-th contributing recording (framing order).
    """
    rec_ids = torch.load(recording_index_path, weights_only=True).tolist()
    pat_ids = torch.load(patient_index_path, weights_only=True).tolist()
    rec2pat: dict[int, int] = {}
    for r, p in zip(rec_ids, pat_ids, strict=True):
        rec2pat.setdefault(r, p)
    patients = sorted(set(pat_ids))
    perm = [patients[i] for i in torch.randperm(len(patients), generator=torch.Generator().manual_seed(seed)).tolist()]
    n_rec = max(rec2pat) + 1
    return [[r for r in range(n_rec) if rec2pat[r] in set(perm[i::k])] for i in range(k)]


def main(args: argparse.Namespace) -> None:
    device = get_device(args.accelerator)
    factor = args.downsample
    win = args.frame_len // factor
    win_stride = args.stride // factor
    wsst = WSST(
        1000, wavelet=args.wavelet, num_voices=args.num_voices, truncate_freq=tuple(args.truncate_freq), stack=True
    )

    print(f"Model: {args.model} | log-dir: {args.log_dir} | device: {device} | rate: {1000 // factor} Hz")
    print("Computing WSST per full recording...")
    dataset = DavidSpringerHSS(args.dataset_path, download=False, framing=False, in_memory=True, verbose=False)

    recordings: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for x, y in dataset:
        if len(x) < args.frame_len:
            continue
        f_ds, l_ds = downsample_time(wsst(x), (y - 1).long(), factor)
        recordings.append((f_ds, l_ds, (y - 1).long()))
        if args.max_recordings and len(recordings) >= args.max_recordings:
            break
    print(f"{len(recordings)} recordings (>= {args.frame_len} samples); feature dim {recordings[0][0].shape[-1]}")
    if args.split_by == "recording":
        rec_folds = recording_test_folds(len(recordings), args.folds, args.seed)
    elif args.split_by == "patient":
        rec_folds = patient_test_folds(args.recording_index, args.patient_index, args.folds, args.seed)
    else:
        rec_folds = None
    if rec_folds is not None:
        print(f"Leakage-free ({args.split_by}): each fold scores only its held-out recordings (~{len(rec_folds[0])})")

    strategies = ["windowed", "stitch", "continuous"]
    per_fold: dict[str, dict[str, list[float]]] = {s: {"macro": [], "s1": []} for s in strategies}
    onset_fold: dict[tuple[str, str, int], list[float]] = {
        (strat, snd, tol): [] for strat in ("stitch", "continuous") for snd in SOUNDS for tol in TOLERANCES_MS
    }

    for i in range(args.folds):
        ckpt = latest_ckpt(args.log_dir, i)
        if ckpt is None:
            print(f"fold {i + 1}: no checkpoint, skipping")
            continue
        net = load_segmenter(args.model, ckpt, device, args.batch_size)
        preds: dict[str, list[torch.Tensor]] = {s: [] for s in strategies}
        win_labels: list[torch.Tensor] = []
        cont_labels: list[torch.Tensor] = []
        onset_rows: dict[tuple[str, str], tuple[list, list]] = {}
        rec_idxs = rec_folds[i] if rec_folds is not None else range(len(recordings))

        with torch.no_grad():
            for ri in rec_idxs:
                f_ds, l_ds, labels_0 = recordings[ri]
                length = f_ds.shape[0]
                starts = window_starts(length, win, win_stride)
                windows = torch.stack([f_ds[s : s + win] for s in starts]).to(device)
                pw = net.decode_valid(windows).cpu()

                preds["windowed"].append(pw.reshape(-1))
                win_labels.append(torch.cat([l_ds[s : s + win] for s in starts]))
                cont_labels.append(l_ds)
                preds["stitch"].append(stitch_central(pw, starts, win, length))
                preds["continuous"].append(net.decode_valid(f_ds[None].to(device)).cpu()[0])

                t2 = length * factor
                ref = labels_0[:t2]
                for strat, seq in (("stitch", preds["stitch"][-1]), ("continuous", preds["continuous"][-1])):
                    up = seq.repeat_interleave(factor)[:t2]
                    for snd, state in SOUNDS.items():
                        p_on, r_on = onset_rows.setdefault((strat, snd), ([], []))
                        p_on.append(onset_lists(up[None], state)[0])
                        r_on.append(onset_lists(ref[None], state)[0])

        wl = torch.cat(win_labels)
        frame_lab = torch.cat(cont_labels)
        per_fold["windowed"]["macro"].append(f1(torch.cat(preds["windowed"]), wl, "macro").item())
        per_fold["windowed"]["s1"].append(f1(torch.cat(preds["windowed"]), wl, "none")[0].item())
        for strat in ("stitch", "continuous"):
            p = torch.cat(preds[strat])
            per_fold[strat]["macro"].append(f1(p, frame_lab, "macro").item())
            per_fold[strat]["s1"].append(f1(p, frame_lab, "none")[0].item())
            for snd in SOUNDS:
                p_on, r_on = onset_rows[(strat, snd)]
                for tol in TOLERANCES_MS:
                    onset_fold[(strat, snd, tol)].append(boundary_f1(p_on, r_on, tol))
        print(f"fold {i + 1}: done")
        del net

    def summ(vals: list[float]) -> str:
        v = torch.tensor(vals)
        return (
            f"{v.mean().item():.4f} ± {v.std(unbiased=True).item():.4f}" if v.numel() > 1 else f"{v.mean().item():.4f}"
        )

    note = "levels are leaked-optimistic" if args.split_by == "all" else f"leakage-free ({args.split_by})"
    print("\n" + "=" * 70)
    print(f"Continuous vs windowed decode (same model + recordings; {note})")
    print("=" * 70)
    print(f"{'strategy':<12} {'macro F1':>20} {'S1 F1':>20}")
    for s in strategies:
        print(f"{s:<12} {summ(per_fold[s]['macro']):>20} {summ(per_fold[s]['s1']):>20}")
    print("\nS1/S2 onset F1 @ 1000 Hz:")
    for snd in SOUNDS:
        for strat in ("stitch", "continuous"):
            row = "  ".join(f"±{tol}={summ(onset_fold[(strat, snd, tol)])}" for tol in TOLERANCES_MS)
            print(f"  {snd} {strat:<11}: {row}")


if __name__ == "__main__":
    main(parse_args())

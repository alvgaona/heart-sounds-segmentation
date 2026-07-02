#!/usr/bin/env python3
"""Train heart sound segmentation model with Semi-Markov CRF (duration-aware)."""

import argparse
import gc

import lightning.pytorch as pl
import torch
import torch.utils.data
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar
from torch.utils.data import DataLoader, TensorDataset

from hss.model.lit_model_semi_crf import LitModelSemiCRF


SEED = 68
FSST_PATH = "data/springer_fsst/springer_fsst.pt"

# Duration priors from actual data, in 1000 Hz frames (~milliseconds):
# S1: mean=132 std=25 | Systole: mean=176 std=56 | S2: mean=96 std=14 | Diastole: mean=347 std=167
DEFAULT_DURATION_MEANS = [130.0, 175.0, 95.0, 350.0]
DEFAULT_DURATION_STDS = [25.0, 55.0, 15.0, 170.0]


def parse_args() -> argparse.Namespace:
    """Parse command-line training options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsst-path", default=FSST_PATH, help="Path to precomputed FSST features (.pt)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for the data split")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    parser.add_argument("--max-epochs", type=int, default=15, help="Maximum training epochs")
    parser.add_argument(
        "--folds",
        type=int,
        default=1,
        help="Number of cross-validation folds. 1 (default) does a single 70/15/15 split; "
        "K>1 runs K-fold CV (test = held-out fold, val carved from the rest) and reports mean±std.",
    )
    parser.add_argument("--patience", type=int, default=6, help="Early-stopping patience (epochs)")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam learning rate")
    parser.add_argument("--max-duration", type=int, default=500, help="Maximum segment duration in frames")
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help=(
            "Temporal downsample factor applied to features (avg-pool) and labels (majority vote). "
            "Duration priors and --max-duration are given in 1000 Hz frames and auto-scaled by this "
            "factor. Use 20 for Springer's 50 Hz resolution."
        ),
    )
    parser.add_argument(
        "--eval-fullres",
        action="store_true",
        help=(
            "After testing, also score at the original (pre-downsample) resolution by upsampling the "
            "decoded predictions back to full length. Removes the boundary-quantization penalty so the "
            "numbers are comparable to metrics measured at 1000 Hz. Test-time only; no training cost."
        ),
    )
    parser.add_argument(
        "--forward-algorithm",
        choices=["sequential", "parallel"],
        default="parallel",
        help="Semi-Markov CRF forward algorithm (parallel required for MPS)",
    )
    parser.add_argument(
        "--duration-means",
        type=float,
        nargs=4,
        metavar=("S1", "SYSTOLE", "S2", "DIASTOLE"),
        default=DEFAULT_DURATION_MEANS,
        help="Initial per-state mean durations (frames)",
    )
    parser.add_argument(
        "--duration-stds",
        type=float,
        nargs=4,
        metavar=("S1", "SYSTOLE", "S2", "DIASTOLE"),
        default=DEFAULT_DURATION_STDS,
        help="Initial per-state duration std devs (frames)",
    )
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes per split")
    parser.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "gpu", "mps"],
        default="cpu",
        help=(
            "Trainer accelerator. Defaults to cpu: the Semi-Markov CRF log-space forward hits an "
            "Apple MPS kernel bug (log_Z overflows to +inf) that CPU does not; use gpu/mps only on CUDA."
        ),
    )
    parser.add_argument(
        "--log-dir", default="lightning_logs_semi_crf", help="Trainer default_root_dir for logs/checkpoints"
    )
    return parser.parse_args()


def downsample_time(features: torch.Tensor, labels: torch.Tensor, factor: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce temporal resolution: average-pool features, majority-vote labels.

    Args:
        features: (N, T, C)
        labels: (N, T)
        factor: pooling window; the tail that doesn't fill a full window is dropped.

    Returns:
        (features_pooled, labels_pooled) with new length T // factor.
    """
    if factor <= 1:
        return features, labels

    n, t, c = features.shape
    t2 = (t // factor) * factor
    new_t = t2 // factor

    # Average-pool features (anti-aliasing) instead of striding, which would alias.
    features = features[:, :t2, :].reshape(n, new_t, factor, c).mean(dim=2)
    # Majority vote per window so boundary frames don't inject label noise.
    labels = labels[:, :t2].reshape(n, new_t, factor).mode(dim=2).values
    return features, labels


def max_segment_length(labels: torch.Tensor) -> int:
    """Longest run of a constant label across all sequences (row starts count as boundaries)."""
    n, t = labels.shape
    flat = labels.reshape(-1)
    change = torch.ones(flat.numel(), dtype=torch.bool)
    change[1:] = flat[1:] != flat[:-1]
    change[::t] = True
    starts = torch.nonzero(change).flatten()
    ends = torch.cat([starts[1:], torch.tensor([flat.numel()])])
    return int((ends - starts).max())


def evaluate_fullres(
    lit_model: "LitModelSemiCRF",
    features: torch.Tensor,
    labels_full: torch.Tensor,
    factor: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Score decoded predictions at the original resolution.

    Decodes at the (downsampled) working resolution, upsamples each predicted frame back by `factor`,
    and compares against the original full-resolution labels. This removes the boundary-quantization
    penalty that inflates per-frame error when metrics are computed on the short downsampled sequence.

    Args:
        lit_model: trained LitModelSemiCRF
        features: (N, T, C) downsampled test features
        labels_full: (N, T*factor) original-resolution test labels (trimmed to a multiple of factor)
        factor: downsample factor used
        batch_size: chunk size for decoding
        device: device to run on

    Returns:
        Dict of macro and per-class accuracy/precision/recall/f1 at full resolution.
    """
    from torchmetrics import MetricCollection
    from torchmetrics.classification import Accuracy, F1Score, Precision, Recall

    num_classes = 4
    t2 = (labels_full.shape[1] // factor) * factor
    labels_full = labels_full[:, :t2]

    def make(average: str) -> MetricCollection:
        return MetricCollection(
            {
                "accuracy": Accuracy(task="multiclass", num_classes=num_classes, average=average),
                "precision": Precision(task="multiclass", num_classes=num_classes, average=average),
                "recall": Recall(task="multiclass", num_classes=num_classes, average=average),
                "f1": F1Score(task="multiclass", num_classes=num_classes, average=average),
            }
        ).to(device)

    macro = make("macro")
    per_class = make("none")

    net = lit_model.model
    net.eval()
    with torch.no_grad():
        for i in range(0, features.shape[0], batch_size):
            x = features[i : i + batch_size].to(device)
            target = labels_full[i : i + batch_size].to(device)  # (b, t2)
            preds = net.decode_valid(x)  # (b, T) guaranteed-valid cardiac cycle
            preds_up = preds.repeat_interleave(factor, dim=1)  # (b, t2)
            macro.update(preds_up, target)
            per_class.update(preds_up, target)

    out = {k: v for k, v in macro.compute().items()}
    out.update({f"{k}_per_class": v for k, v in per_class.compute().items()})
    return out


def get_device(accelerator: str) -> tuple[torch.device, str]:
    """Resolve the torch device and Lightning accelerator string from the requested accelerator."""
    if accelerator == "cpu":
        return torch.device("cpu"), "cpu"
    if accelerator in ("gpu", "mps"):
        if torch.cuda.is_available():
            return torch.device("cuda"), "gpu"
        if torch.backends.mps.is_available():
            return torch.device("mps"), "mps"
        return torch.device("cpu"), "cpu"
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda"), "gpu"
    if torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def kfold_indices(n: int, k: int, seed: int) -> list[tuple[list[int], list[int], list[int]]]:
    """Build (train, val, test) index splits for k-fold CV.

    Test is the held-out fold; val is 15% of the remaining data (for early stopping); train is the
    rest. The 15% carve keeps train non-empty for any k >= 2. Deterministic given the seed.
    """
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    folds = [perm[i::k] for i in range(k)]  # round-robin partition, disjoint and covering
    splits = []
    for i in range(k):
        test_idx = folds[i]
        rest = [idx for j, f in enumerate(folds) if j != i for idx in f]
        val_size = int(0.15 * len(rest))
        val_idx, train_idx = rest[:val_size], rest[val_size:]
        splits.append((train_idx, val_idx, test_idx))
    return splits


def run_split(
    features: torch.Tensor,
    labels: torch.Tensor,
    labels_full: torch.Tensor | None,
    split: tuple[list[int], list[int], list[int]],
    args: argparse.Namespace,
    device: torch.device,
    accelerator: str,
    duration_means: list[float],
    duration_stds: list[float],
    max_duration: int,
    log_dir: str,
) -> tuple[dict[str, float], dict[str, torch.Tensor] | None]:
    """Train and test one train/val/test split; return the test metrics (and matched-res metrics)."""
    train_idx, val_idx, test_idx = split
    dataset = TensorDataset(features, labels)

    def loader(idx: list[int], shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(
            torch.utils.data.Subset(dataset, idx),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            drop_last=drop_last,
            persistent_workers=False,
        )

    model = LitModelSemiCRF(
        input_size=44,
        batch_size=args.batch_size,
        device=device,
        max_duration=max_duration,
        duration_means=duration_means,
        duration_stds=duration_stds,
        forward_algorithm=args.forward_algorithm,
        lr=args.lr,
    )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        gradient_clip_val=1,
        gradient_clip_algorithm="norm",
        callbacks=[EarlyStopping("val_loss", patience=args.patience, check_finite=True), RichProgressBar()],
        default_root_dir=log_dir,
    )
    # Train batches drop the partial tail; val/test keep every sample (and val must be non-empty
    # so EarlyStopping always has val_loss).
    trainer.fit(model, loader(train_idx, True, drop_last=True), loader(val_idx, False, drop_last=False))
    test_results = trainer.test(dataloaders=loader(test_idx, False, drop_last=False), ckpt_path="best")[0]

    fullres = None
    if labels_full is not None:
        fullres = evaluate_fullres(
            model.to(device), features[test_idx], labels_full[test_idx], args.downsample, args.batch_size, device
        )

    # Release the model/trainer so GPU memory doesn't accumulate across CV folds.
    del model, trainer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return test_results, fullres


def _print_metrics(test_results: dict[str, float], fullres: dict[str, torch.Tensor] | None, scale: int) -> None:
    print("\n" + "=" * 60)
    print(f"SEMI-MARKOV CRF MODEL TEST RESULTS (@ {1000 // scale} Hz)")
    print("=" * 60)
    for key, value in sorted(test_results.items()):
        print(f"{key}: {value:.4f}")
    if fullres is not None:
        state_names = ["S1", "Systole", "S2", "Diastole"]
        print("\nMATCHED-RESOLUTION TEST RESULTS (decoded, upsampled to 1000 Hz)")
        for key in ("accuracy", "precision", "recall", "f1"):
            print(f"fullres_{key}: {float(fullres[key]):.4f}")
        for key in ("accuracy", "precision", "recall", "f1"):
            per_class = ", ".join(f"{n}={v:.4f}" for n, v in zip(state_names, fullres[f"{key}_per_class"], strict=True))
            print(f"fullres_{key}_per_class: {per_class}")


def _print_cv_summary(all_results: list[dict[str, float]], k: int) -> None:
    print("\n" + "=" * 60)
    print(f"{k}-FOLD CV SUMMARY (mean ± std across folds)")
    print("=" * 60)
    keys = sorted(set().union(*(r.keys() for r in all_results)))
    for key in keys:
        vals = torch.tensor([r[key] for r in all_results if key in r])
        std = vals.std(unbiased=True).item() if vals.numel() > 1 else 0.0
        print(f"{key}: {vals.mean().item():.4f} ± {std:.4f}")


def main(args: argparse.Namespace) -> None:
    device, accelerator = get_device(args.accelerator)
    print(f"Using device: {device} (accelerator: {accelerator})")
    print("Training with Semi-Markov CRF model (duration-aware)")

    # Load precomputed FSST features
    print(f"Loading precomputed features from {args.fsst_path}...")
    data = torch.load(args.fsst_path, weights_only=True, mmap=args.downsample > 1)
    features = data["features"]
    labels = data["labels"]
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

    # Keep original-resolution labels for matched-resolution evaluation before downsampling.
    labels_full = labels.clone() if (args.eval_fullres and args.downsample > 1) else None

    if args.downsample > 1:
        features, labels = downsample_time(features, labels, args.downsample)
        print(f"Downsampled x{args.downsample}: features {tuple(features.shape)}, labels {tuple(labels.shape)}")

    n = len(features)

    # Duration priors and max_duration are specified in 1000 Hz frames; scale to the
    # working resolution so they stay physically meaningful after downsampling.
    scale = args.downsample
    duration_means = [m / scale for m in args.duration_means]
    duration_stds = [s / scale for s in args.duration_stds]
    max_duration = max(1, round(args.max_duration / scale))
    print(f"Duration priors (frames @ {1000 // scale} Hz): means={duration_means}, stds={duration_stds}")

    # Every ground-truth segment must be representable (duration <= max_duration), otherwise the
    # numerator scores paths the partition function excludes and training diverges. For downsampled
    # runs the sequence is short, so raise max_duration to cover the data (segment cost is ~D-independent).
    if scale > 1:
        seq_len = features.shape[1]
        data_max = max_segment_length(labels)
        covered = min(seq_len, max(max_duration, data_max))
        if covered != max_duration:
            print(f"Raising max_duration {max_duration} -> {covered} to cover longest segment ({data_max} frames)")
        max_duration = covered
    print(f"Max duration: {max_duration} frames")

    common = (features, labels, labels_full)
    params = (duration_means, duration_stds, max_duration)

    if args.folds <= 1:
        # Single 70/15/15 split (test and val are each 15%).
        test_size = int(0.15 * n)
        val_size = int(0.15 * n)
        subsets = torch.utils.data.random_split(
            TensorDataset(features, labels),
            [n - test_size - val_size, val_size, test_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
        split = (subsets[0].indices, subsets[1].indices, subsets[2].indices)
        test_results, fullres = run_split(*common, split, args, device, accelerator, *params, args.log_dir)
        _print_metrics(test_results, fullres, scale)
        return

    # K-fold cross-validation.
    all_results: list[dict[str, float]] = []
    for i, split in enumerate(kfold_indices(n, args.folds, args.seed)):
        print("\n" + "#" * 60)
        print(f"# FOLD {i + 1}/{args.folds}")
        print("#" * 60)
        test_results, fullres = run_split(
            *common, split, args, device, accelerator, *params, f"{args.log_dir}/fold_{i}"
        )
        _print_metrics(test_results, fullres, scale)
        all_results.append(test_results)

    _print_cv_summary(all_results, args.folds)


if __name__ == "__main__":
    main(parse_args())

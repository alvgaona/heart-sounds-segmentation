#!/usr/bin/env python3
"""Train heart sound segmentation model with (Bi)LSTM + CRF layer."""

import argparse
import gc

import lightning.pytorch as pl
import torch
import torch.utils.data
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar
from torch.utils.data import DataLoader, TensorDataset

from hss.model.boundary_loss import BoundaryLossConfig
from hss.model.lit_model_crf import LitModelCRF


SEED = 68
FSST_PATH = "data/springer_fsst/springer_fsst.pt"


def parse_args() -> argparse.Namespace:
    """Parse command-line training options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsst-path", default=FSST_PATH, help="Path to precomputed FSST features (.pt)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for the data split")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    parser.add_argument("--max-epochs", type=int, default=15, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=6, help="Early-stopping patience (epochs)")
    parser.add_argument(
        "--folds",
        type=int,
        default=1,
        help="Number of cross-validation folds. 1 (default) does a single 70/15/15 split; "
        "K>1 runs K-fold CV (test = held-out fold, val carved from the rest) and reports mean±std.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help=(
            "Temporal downsample factor applied to features (avg-pool) and labels (majority vote). "
            "Use 20 for Springer's 50 Hz resolution (~20x faster than full 1000 Hz, and the "
            "apples-to-apples resolution for comparing against the 50 Hz semi-Markov CRF)."
        ),
    )
    parser.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "gpu", "mps"],
        default="auto",
        help="Trainer accelerator (auto picks CUDA/MPS/CPU). The linear-chain CRF runs fine on MPS.",
    )
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes per split")
    parser.add_argument("--log-dir", default="lightning_logs_crf", help="Trainer default_root_dir for logs/checkpoints")
    parser.add_argument("--lr", type=float, default=0.01, help="Adam learning rate")
    parser.add_argument(
        "--boundary-loss",
        action="store_true",
        help="Add the S1-focused boundary-aware auxiliary loss (weighted per-frame CE on emissions) "
        "to the CRF NLL. Targets missed/spurious S1 detections. Use a separate --log-dir to avoid "
        "overwriting the baseline checkpoints.",
    )
    parser.add_argument(
        "--aux-lambda",
        type=float,
        default=5.0,
        help="Weight of the aux CE relative to the CRF NLL. The CRF NLL is a per-sequence sum (~9) while the "
        "aux is a per-frame weighted mean (~0.43), so lambda~5 makes the aux ~25%% of the loss (calibrated on "
        "a trained baseline); lambda<1 is effectively inert.",
    )
    parser.add_argument(
        "--boundary-weight", type=float, default=2.0, help="Loss multiplier for frames near a GT transition"
    )
    parser.add_argument(
        "--boundary-window", type=int, default=2, help="Half-width (frames) of the boundary emphasis region"
    )
    parser.add_argument("--s1-weight", type=float, default=2.0, help="Class weight for S1 (class 0) in the aux CE")
    return parser.parse_args()


def build_boundary_cfg(args: argparse.Namespace) -> BoundaryLossConfig:
    """Construct the boundary-loss config from CLI args (disabled unless --boundary-loss)."""
    if not args.boundary_loss:
        return BoundaryLossConfig(aux_lambda=0.0)
    return BoundaryLossConfig(
        aux_lambda=args.aux_lambda,
        boundary_weight=args.boundary_weight,
        boundary_window=args.boundary_window,
        class_weights=(args.s1_weight, 1.0, 1.0, 1.0),
    )


def downsample_time(features: torch.Tensor, labels: torch.Tensor, factor: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce temporal resolution: average-pool features (anti-aliasing), majority-vote labels.

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
    features = features[:, :t2, :].reshape(n, new_t, factor, c).mean(dim=2)
    labels = labels[:, :t2].reshape(n, new_t, factor).mode(dim=2).values
    return features, labels


def get_device(accelerator: str) -> tuple[torch.device, str]:
    """Resolve the torch device and Lightning accelerator string from the requested accelerator."""
    if accelerator == "cpu":
        return torch.device("cpu"), "cpu"
    want_gpu = accelerator in ("auto", "gpu", "mps")
    if want_gpu and torch.cuda.is_available():
        return torch.device("cuda"), "gpu"
    if want_gpu and torch.backends.mps.is_available():
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
    split: tuple[list[int], list[int], list[int]],
    args: argparse.Namespace,
    device: torch.device,
    accelerator: str,
    log_dir: str,
) -> dict[str, float]:
    """Train and test one train/val/test split; return the test metrics."""
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

    model = LitModelCRF(
        input_size=features.shape[-1],
        batch_size=args.batch_size,
        device=device,
        lr=args.lr,
        boundary_cfg=build_boundary_cfg(args),
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

    del model, trainer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return test_results


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
    print("Training with (Bi)LSTM + CRF model")
    if args.boundary_loss:
        print(
            f"Boundary-aware aux loss ON: lambda={args.aux_lambda}, boundary_weight={args.boundary_weight}, "
            f"window={args.boundary_window}, s1_weight={args.s1_weight}"
        )

    print(f"Loading precomputed features from {args.fsst_path}...")
    data = torch.load(args.fsst_path, weights_only=True, mmap=args.downsample > 1)
    features = data["features"]
    labels = data["labels"]
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

    if args.downsample > 1:
        features, labels = downsample_time(features, labels, args.downsample)
        print(f"Downsampled x{args.downsample}: features {tuple(features.shape)}, labels {tuple(labels.shape)}")

    n = len(features)

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
        test_results = run_split(features, labels, split, args, device, accelerator, args.log_dir)
        print("\n" + "=" * 60)
        print(f"CRF MODEL TEST RESULTS (@ {1000 // args.downsample} Hz)")
        print("=" * 60)
        for key, value in sorted(test_results.items()):
            print(f"{key}: {value:.4f}")
        return

    # K-fold cross-validation.
    all_results: list[dict[str, float]] = []
    for i, split in enumerate(kfold_indices(n, args.folds, args.seed)):
        print("\n" + "#" * 60)
        print(f"# FOLD {i + 1}/{args.folds}")
        print("#" * 60)
        test_results = run_split(features, labels, split, args, device, accelerator, f"{args.log_dir}/fold_{i}")
        for key, value in sorted(test_results.items()):
            print(f"{key}: {value:.4f}")
        all_results.append(test_results)

    _print_cv_summary(all_results, args.folds)


if __name__ == "__main__":
    main(parse_args())

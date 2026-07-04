#!/usr/bin/env python3
"""Train heart sound segmentation model with (Bi)LSTM + CrossEntropy.

The original model in this repo: a 2-layer bidirectional LSTM with a per-frame CrossEntropy loss
(no CRF). Mirrors scripts/training/train_crf.py — argparse CLI, optional temporal downsampling, and k-fold CV
on precomputed FSST features — so the plain LSTM is comparable to the CRF variants under one protocol.
"""

import argparse
import gc

import lightning.pytorch as pl
import torch
import torch.utils.data
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar
from torch.utils.data import DataLoader, TensorDataset

from hss.model.lit_model import LitModel


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
        "--split-by",
        choices=["frame", "recording", "patient"],
        default="frame",
        help="CV split granularity. 'frame' (default) is the original behavior. 'recording'/'patient' group "
        "each recording's/patient's windows into one fold — no leakage. 'patient' matches Springer 2016 (135 "
        "patients). Needs the corresponding index and --folds >= 2.",
    )
    parser.add_argument("--recording-index", default="data/recording_ids.pt", help="frame->recording id tensor")
    parser.add_argument("--patient-index", default="data/patient_ids.pt", help="frame->patient id tensor")
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help=(
            "Temporal downsample factor applied to features (avg-pool) and labels (majority vote). "
            "Use 20 for Springer's 50 Hz resolution (~20x faster than full 1000 Hz, and the "
            "apples-to-apples resolution for comparing against the CRF variants)."
        ),
    )
    parser.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "gpu", "mps"],
        default="auto",
        help="Trainer accelerator (auto picks CUDA/MPS/CPU). The LSTM runs fine on MPS.",
    )
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes per split")
    parser.add_argument(
        "--log-dir", default="lightning_logs_lstm", help="Trainer default_root_dir for logs/checkpoints"
    )
    return parser.parse_args()


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


def grouped_kfold_indices(group_ids: torch.Tensor, k: int, seed: int) -> list[tuple[list[int], list[int], list[int]]]:
    """K-fold splits that keep every group's frames in a single fold (no leakage across the group boundary).

    A group is a recording or a patient. Robust to non-contiguous ids (e.g. patient numbers 1..135); for
    contiguous 0..N-1 it matches the frame-level perm.
    """
    groups = torch.unique(group_ids).tolist()
    frames_by_group: dict[int, list[int]] = {g: [] for g in groups}
    for frame_idx, g in enumerate(group_ids.tolist()):
        frames_by_group[g].append(frame_idx)

    perm = [groups[i] for i in torch.randperm(len(groups), generator=torch.Generator().manual_seed(seed)).tolist()]
    folds = [perm[i::k] for i in range(k)]

    def to_frames(gs: list[int]) -> list[int]:
        return [fi for g in gs for fi in frames_by_group[g]]

    splits = []
    for i in range(k):
        test = folds[i]
        rest = [g for j, f in enumerate(folds) if j != i for g in f]
        val_size = int(0.15 * len(rest))
        splits.append((to_frames(rest[val_size:]), to_frames(rest[:val_size]), to_frames(test)))
    return splits


def build_fold_splits(n: int, args: argparse.Namespace) -> list[tuple[list[int], list[int], list[int]]]:
    """Frame-level (default) or grouped (recording/patient) k-fold splits."""
    if args.split_by == "frame":
        return kfold_indices(n, args.folds, args.seed)
    index_path = args.recording_index if args.split_by == "recording" else args.patient_index
    group_ids = torch.load(index_path, weights_only=True)
    if len(group_ids) != n:
        raise ValueError(f"{args.split_by} index has {len(group_ids)} frames, features have {n}")
    print(f"{args.split_by.capitalize()}-level splits: {len(torch.unique(group_ids))} groups across {args.folds} folds")
    return grouped_kfold_indices(group_ids, args.folds, args.seed)


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

    model = LitModel(input_size=features.shape[-1], batch_size=args.batch_size, device=device)
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
    print("Training with (Bi)LSTM + CrossEntropy model")

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
        print(f"LSTM MODEL TEST RESULTS (@ {1000 // args.downsample} Hz)")
        print("=" * 60)
        for key, value in sorted(test_results.items()):
            print(f"{key}: {value:.4f}")
        return

    # K-fold cross-validation.
    all_results: list[dict[str, float]] = []
    for i, split in enumerate(build_fold_splits(n, args)):
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

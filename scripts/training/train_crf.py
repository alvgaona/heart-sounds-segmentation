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
        "--split-by",
        choices=["frame", "recording", "patient"],
        default="frame",
        help="CV split granularity. 'frame' (default) splits the flat frame stack (original behavior). "
        "'recording'/'patient' group each recording's/patient's windows into one fold — no leakage. "
        "'patient' matches Springer 2016 (135 patients). Needs the corresponding index and --folds >= 2.",
    )
    parser.add_argument(
        "--recording-index",
        default="data/recording_ids.pt",
        help="frame->recording id tensor for --split-by recording (build with build_recording_index.py)",
    )
    parser.add_argument(
        "--patient-index",
        default="data/patient_ids.pt",
        help="frame->patient id tensor for --split-by patient (135 Springer patients)",
    )
    parser.add_argument(
        "--extra-train-path",
        default=None,
        help="Feature .pt (same dim) ALWAYS added to the train set of every fold, never validated/tested. "
        "For joint cross-dataset training (e.g. --fsst-path CirCor, --extra-train-path Springer). CV folds are "
        "still defined only over --fsst-path, so the test set stays the held-out --fsst-path subset.",
    )
    parser.add_argument(
        "--extra-downsample",
        type=int,
        default=1,
        help="Avg-pool factor for --extra-train-path (20 for 1000 Hz Springer)",
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
        "--arch",
        choices=["bilstm", "xlstm"],
        default="bilstm",
        help="Recurrent emitter: 2-layer BiLSTM (default) or the vendored xLSTM (Experiment A)",
    )
    parser.add_argument("--xlstm-hidden", type=int, default=240, help="Per-direction hidden width (xLSTM/BiLSTM)")
    parser.add_argument("--xlstm-heads", type=int, default=4, help="mLSTM heads per layer (must divide hidden)")
    parser.add_argument("--xlstm-layers", type=int, default=2, help="Number of stacked bidirectional mLSTM layers")
    parser.add_argument(
        "--causal", action="store_true", help="xLSTM only: unidirectional (streamable) emitter instead of bidirectional"
    )
    parser.add_argument(
        "--phase", action="store_true", help="xLSTM only: phase-clock mLSTM with a cardiac-phase inductive bias (Exp C)"
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Fraction of TRAIN patients to keep for a data-efficiency sweep (val/test stay full). "
        "Patient-level, nested, seeded (use with --split-by patient).",
    )
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


def kfold_indices(
    n: int, k: int, seed: int, train_fraction: float = 1.0
) -> list[tuple[list[int], list[int], list[int]]]:
    """Build (train, val, test) index splits for k-fold CV.

    Test is the held-out fold; val is 15% of the remaining data (for early stopping); train is the
    rest. The 15% carve keeps train non-empty for any k >= 2. Deterministic given the seed.
    ``train_fraction`` < 1 keeps a nested seeded prefix of train (frame-level; use the grouped/patient
    variant for a leakage-safe data-efficiency sweep).
    """
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    folds = [perm[i::k] for i in range(k)]  # round-robin partition, disjoint and covering
    splits = []
    for i in range(k):
        test_idx = folds[i]
        rest = [idx for j, f in enumerate(folds) if j != i for idx in f]
        val_size = int(0.15 * len(rest))
        val_idx, train_idx = rest[:val_size], rest[val_size:]
        if train_fraction < 1.0:
            train_idx = train_idx[: max(1, round(train_fraction * len(train_idx)))]
        splits.append((train_idx, val_idx, test_idx))
    return splits


def grouped_kfold_indices(
    group_ids: torch.Tensor, k: int, seed: int, train_fraction: float = 1.0
) -> list[tuple[list[int], list[int], list[int]]]:
    """K-fold splits that keep every group's frames in a single fold (no leakage across the group boundary).

    A group is a recording or a patient. Splits the GROUPS round-robin (like kfold_indices splits frames),
    carves 15% of the remaining groups for val, then maps group ids back to their frame indices. Robust to
    non-contiguous ids (e.g. patient numbers 1..135); for contiguous 0..N-1 it matches the frame-level perm.
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
        val_groups, train_groups = rest[:val_size], rest[val_size:]
        if train_fraction < 1.0:
            # Data-efficiency sweep: keep a nested, seeded prefix of the (already-shuffled) train patients.
            # val/test stay full, so only the training-label budget shrinks. 0.05 ⊂ 0.1 ⊂ ... (nested).
            train_groups = train_groups[: max(1, round(train_fraction * len(train_groups)))]
        splits.append((to_frames(train_groups), to_frames(val_groups), to_frames(test)))
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
        arch=args.arch,
        hidden_size=args.xlstm_hidden,
        num_heads=args.xlstm_heads,
        num_layers=args.xlstm_layers,
        bidirectional=not args.causal,
        phase=args.phase,
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
    emitter = "xLSTM" if args.arch == "xlstm" else "BiLSTM"
    print(f"Training with {emitter} + CRF model")
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

    # Joint training: append an extra dataset (e.g. Springer) to every fold's TRAIN set. CV folds below are
    # defined over the first n frames only, so the test set stays the held-out --fsst-path subset.
    extra_idx: list[int] = []
    if args.extra_train_path:
        extra = torch.load(args.extra_train_path, weights_only=True, mmap=args.extra_downsample > 1)
        ef, el = extra["features"], extra["labels"]
        if args.extra_downsample > 1:
            ef, el = downsample_time(ef, el, args.extra_downsample)
        if ef.shape[-1] != features.shape[-1]:
            raise ValueError(f"extra feature dim {ef.shape[-1]} != main {features.shape[-1]}")
        extra_idx = list(range(n, n + len(ef)))
        features = torch.cat([features, ef])
        labels = torch.cat([labels, el])
        print(f"Joint: +{len(ef)} extra train frames from {args.extra_train_path} (always in train)")

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
    if args.split_by in ("recording", "patient"):
        index_path = args.recording_index if args.split_by == "recording" else args.patient_index
        group_ids = torch.load(index_path, weights_only=True)
        if len(group_ids) != n:
            raise ValueError(f"{args.split_by} index has {len(group_ids)} frames, features have {n}")
        fold_splits = grouped_kfold_indices(group_ids, args.folds, args.seed, args.train_fraction)
        n_groups = len(torch.unique(group_ids))
        print(f"{args.split_by.capitalize()}-level splits: {n_groups} groups across {args.folds} folds")
    else:
        fold_splits = kfold_indices(n, args.folds, args.seed, args.train_fraction)

    all_results: list[dict[str, float]] = []
    for i, split in enumerate(fold_splits):
        print("\n" + "#" * 60)
        print(f"# FOLD {i + 1}/{args.folds}")
        print("#" * 60)
        split = (list(split[0]) + extra_idx, split[1], split[2])  # extra data is train-only
        test_results = run_split(features, labels, split, args, device, accelerator, f"{args.log_dir}/fold_{i}")
        for key, value in sorted(test_results.items()):
            print(f"{key}: {value:.4f}")
        all_results.append(test_results)

    _print_cv_summary(all_results, args.folds)


if __name__ == "__main__":
    main(parse_args())

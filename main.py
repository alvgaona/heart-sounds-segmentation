import os

import lightning.pytorch as pl
import scipy
import torch
import torch.utils.data
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms

from hss.datasets.heart_sounds import DavidSpringerHSS
from hss.model.lit_model import LitModel
from hss.transforms import FSST


def get_device() -> tuple[torch.device, str]:
    """Get the best available device and accelerator."""
    if torch.cuda.is_available():
        return torch.device("cuda"), "gpu"
    elif torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def main() -> None:
    device, accelerator = get_device()
    print(f"Using device: {device} (accelerator: {accelerator})")

    transform = transforms.Compose(
        (
            FSST(
                1000,
                window=scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=False),
                truncate_freq=(25, 200),
                stack=True,
            ),
        )
    )

    hss_dataset = DavidSpringerHSS(
        "resources/data",
        download=True,
        framing=True,
        in_memory=True,
        transform=transform,
    )

    batch_size = 50

    # First split dataset into train+val and test sets
    test_size = int(0.15 * len(hss_dataset))
    train_val_size = len(hss_dataset) - test_size

    train_val_dataset, test_dataset = torch.utils.data.random_split(
        hss_dataset, [train_val_size, test_size], generator=torch.Generator().manual_seed(68)
    )

    # Now do k-fold cross validation on the train+val portion
    n_splits = 10
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=68)

    # Initialize lists to store metrics for each fold
    fold_metrics = [
        {
            "accuracy": torch.zeros(n_splits),
            "precision": torch.zeros(n_splits),
            "recall": torch.zeros(n_splits),
            "f1": torch.zeros(n_splits),
            "MulticlassAUROC": torch.zeros(n_splits),
        }
        for _ in range(4)
    ]

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(range(train_val_size))):
        # Create samplers for data loading
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        # Create data loaders
        train_loader = DataLoader(
            train_val_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=os.cpu_count() or 4,
            drop_last=True,
            persistent_workers=True,
        )

        val_loader = DataLoader(
            train_val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=os.cpu_count() or 4,
            drop_last=True,
            persistent_workers=True,
        )

        # Initialize model and training
        model = LitModel(input_size=44, batch_size=batch_size, device=device)
        early_stopping = EarlyStopping("val_loss", patience=6, check_finite=True)

        trainer = pl.Trainer(
            max_epochs=15,
            accelerator=accelerator,
            gradient_clip_val=1,
            gradient_clip_algorithm="norm",
            callbacks=[early_stopping, RichProgressBar()],
        )

        # Train and validate for this fold
        trainer.fit(model, train_loader, val_loader)

        # Create test loader from held-out test set
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=os.cpu_count() or 4,
            drop_last=True,
        )

        # Test the model
        test_results = trainer.test(dataloaders=test_loader, ckpt_path="best")[0]

        # Loop through classes and metrics
        metrics = ["accuracy", "precision", "recall", "f1", "MulticlassAUROC"]

        for metric in metrics:
            for i in range(4):
                metric_key = f"test_{metric}_{i}"
                fold_metrics[i][metric][fold_idx] = test_results[metric_key]

    for i, metrics in enumerate(fold_metrics):
        print(f"Class {i}")
        print("---")
        print(f"Accuracy: {torch.mean(metrics['accuracy'])}")
        print(f"Precision: {torch.mean(metrics['precision'])}")
        print(f"Recall: {torch.mean(metrics['recall'])}")
        print(f"F1: {torch.mean(metrics['f1'])}")
        print(f"AUROC: {torch.mean(metrics['MulticlassAUROC'])}\n")


if __name__ == "__main__":
    main()

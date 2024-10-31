from typing import Tuple

import lightning.pytorch as pl
import scipy
import torch
import torch.utils.data
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from sklearn.model_selection import KFold
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchmetrics import MetricCollection
from torchmetrics.classification import AUROC, Accuracy, F1Score, Precision, Recall
from torchvision import transforms

from hss.datasets.heart_sounds import DavidSpringerHSS
from hss.model.segmenter import HeartSoundSegmenter
from hss.transforms import FSST


class LitModel(pl.LightningModule):
    def __init__(self, input_size: int, batch_size: int, device: torch.device) -> None:
        super().__init__()
        self.model = HeartSoundSegmenter(
            input_size=input_size,
            batch_size=batch_size,
            device=device,
            dtype=torch.float32,
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        num_classes = 4

        self.train_metrics = MetricCollection(
            {
                "accuracy": Accuracy(task="multiclass", num_classes=num_classes),
                "precision": Precision(task="multiclass", average="macro", num_classes=num_classes),
                "recall": Recall(task="multiclass", average="macro", num_classes=num_classes),
                "f1": F1Score(task="multiclass", num_classes=num_classes),
            },
            prefix="train_",
        )

        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")
        self.test_metrics.add_metrics(AUROC(task="multiclass", num_classes=num_classes, average="macro"))

    def on_train_epoch_end(self) -> None:
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.val_metrics.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        outputs = self(x).permute((0, 2, 1))
        loss = self.loss_fn(outputs, y)

        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self.log_dict(self.train_metrics(outputs, y), prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        outputs = self(x).permute((0, 2, 1))
        loss = self.loss_fn(outputs, y)

        self.log("val_loss", loss, prog_bar=True, on_step=True)
        self.log_dict(self.val_metrics(outputs, y), prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        outputs = self(x).permute((0, 2, 1))
        loss = self.loss_fn(outputs, y)

        self.log("test_loss", loss, prog_bar=True, on_step=True)
        self.log_dict(self.test_metrics(outputs, y), prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = Adam(self.parameters(), lr=0.01)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9**epoch)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        (
            # Resample(35500),
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
    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=68)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(train_val_size))):
        # Create samplers for data loading
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        # Create data loaders
        train_loader = DataLoader(
            train_val_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=19, drop_last=True
        )

        val_loader = DataLoader(
            train_val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=19, drop_last=True
        )

        # Initialize model and training
        model = LitModel(input_size=44, batch_size=batch_size, device=device)
        early_stopping = EarlyStopping("val_loss", patience=6, check_finite=True)

        trainer = pl.Trainer(
            max_epochs=10,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            gradient_clip_val=1,
            gradient_clip_algorithm="norm",
            callbacks=[early_stopping, RichProgressBar()],
        )

        # Train and validate for this fold
        trainer.fit(model, train_loader, val_loader)

        # Optional: Save fold results
        print(f"Fold {fold} completed")

        # Create test loader from held-out test set
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=19, drop_last=True)

        # Test the model
        trainer.test(dataloaders=test_loader, ckpt_path="best")


if __name__ == "__main__":
    main()

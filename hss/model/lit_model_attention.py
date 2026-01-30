"""PyTorch Lightning module for heart sound segmentation with attention."""

from typing import Tuple

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import MetricCollection
from torchmetrics.classification import AUROC, Accuracy, F1Score, Precision, Recall

from hss.model.segmenter_attention import HeartSoundSegmenterAttention


class LitModelAttention(pl.LightningModule):
    """Lightning module using attention-based segmenter."""

    def __init__(
        self,
        input_size: int,
        batch_size: int,
        device: torch.device,
        num_attention_heads: int = 8,
    ) -> None:
        super().__init__()
        self.model = HeartSoundSegmenterAttention(
            input_size=input_size,
            batch_size=batch_size,
            num_attention_heads=num_attention_heads,
            device=device,
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        num_classes = 4

        self.train_metrics_per_class = MetricCollection(
            {
                "accuracy": Accuracy(task="multiclass", average=None, num_classes=num_classes),
                "precision": Precision(task="multiclass", average=None, num_classes=num_classes),
                "recall": Recall(task="multiclass", average=None, num_classes=num_classes),
                "f1": F1Score(task="multiclass", average=None, num_classes=num_classes),
            },
            prefix="train_per_class_",
        )

        self.val_metrics_per_class = self.train_metrics_per_class.clone(prefix="val_")
        self.test_metrics_per_class = self.train_metrics_per_class.clone(prefix="test_")
        self.test_metrics_per_class.add_metrics(AUROC(task="multiclass", average=None, num_classes=num_classes))

        self.train_metrics = MetricCollection(
            {
                "accuracy": Accuracy(task="multiclass", average="macro", num_classes=num_classes),
                "precision": Precision(task="multiclass", average="macro", num_classes=num_classes),
                "recall": Recall(task="multiclass", average="macro", num_classes=num_classes),
                "f1": F1Score(task="multiclass", average="macro", num_classes=num_classes),
            },
            prefix="train_",
        )

        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")
        self.test_metrics.add_metrics(AUROC(task="multiclass", average="macro", num_classes=num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x).permute((0, 2, 1))
        loss = self.loss_fn(logits, y)

        metrics_per_class = self.train_metrics_per_class(logits, y)
        self.train_metrics_per_class.reset()

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log_dict(self.train_metrics(logits, y), prog_bar=True, on_step=True, on_epoch=True)

        for metric_name, metric_values in metrics_per_class.items():
            for i, v in enumerate(metric_values):
                self.log(f"{metric_name}_{i}", v)

        return loss

    def on_train_epoch_end(self) -> None:
        self.train_metrics_per_class.reset()
        self.train_metrics.reset()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x).permute((0, 2, 1))
        loss = self.loss_fn(logits, y)

        metrics_per_class = self.val_metrics_per_class(logits, y)
        self.val_metrics_per_class.reset()

        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log_dict(self.val_metrics(logits, y), prog_bar=True, on_step=False, on_epoch=True)

        for metric_name, metric_values in metrics_per_class.items():
            for i, v in enumerate(metric_values):
                self.log(f"{metric_name}_{i}", v)

        return loss

    def on_validation_epoch_end(self) -> None:
        self.val_metrics_per_class.reset()
        self.val_metrics.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x).permute((0, 2, 1))
        loss = self.loss_fn(logits, y)

        metrics_per_class = self.test_metrics_per_class(logits, y)
        self.test_metrics_per_class.reset()

        self.log("test_loss", loss)
        self.log_dict(self.test_metrics(logits, y))

        for metric_name, metric_values in metrics_per_class.items():
            for i, v in enumerate(metric_values):
                self.log(f"{metric_name}_{i}", v)

        return loss

    def on_test_epoch_end(self) -> None:
        self.test_metrics_per_class.reset()
        self.test_metrics.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = Adam(self.parameters(), lr=0.01)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9**epoch)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

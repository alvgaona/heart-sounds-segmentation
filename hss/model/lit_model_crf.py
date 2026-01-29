"""PyTorch Lightning module for heart sound segmentation with CRF."""

from typing import Tuple

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import MetricCollection
from torchmetrics.classification import AUROC, Accuracy, F1Score, Precision, Recall

from hss.model.segmenter_crf import HeartSoundSegmenterCRF


class LitModelCRF(pl.LightningModule):
    """Lightning module using CRF-based segmenter."""

    def __init__(
        self,
        input_size: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.model = HeartSoundSegmenterCRF(
            input_size=input_size,
            batch_size=batch_size,
            device=device,
        )
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

    def _decode_to_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Decode CRF and convert to one-hot logits for metrics."""
        decoded = self.model.decode(x)
        batch_size, seq_len = len(decoded), len(decoded[0])

        # Create one-hot logits from decoded sequences
        logits = torch.zeros(batch_size, 4, seq_len, device=x.device)
        for b, seq in enumerate(decoded):
            for t, tag in enumerate(seq):
                logits[b, tag, t] = 10.0  # High logit for predicted class

        return logits

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch

        # CRF loss (negative log-likelihood)
        loss = self.model.loss(x, y)

        # Decode for metrics
        logits = self._decode_to_logits(x)

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

        # CRF loss
        loss = self.model.loss(x, y)

        # Decode for metrics
        logits = self._decode_to_logits(x)

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

        # CRF loss
        loss = self.model.loss(x, y)

        # Decode for metrics
        logits = self._decode_to_logits(x)

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

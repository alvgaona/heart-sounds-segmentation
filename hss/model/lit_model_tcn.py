"""PyTorch Lightning module for heart sound segmentation with TCN + CRF."""

from typing import Tuple

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MetricCollection
from torchmetrics.classification import AUROC, Accuracy, F1Score, Precision, Recall

from hss.model.tcn import HeartSoundSegmenterBiTCN, HeartSoundSegmenterTCN


class LitModelTCN(pl.LightningModule):
    """Lightning module for TCN + CRF heart sound segmentation.

    Args:
        input_size: Number of input features (44 for FSST)
        hidden_size: TCN hidden dimension
        num_layers: Number of TCN blocks
        kernel_size: Convolution kernel size
        dropout: Dropout probability
        bidirectional: Whether to use bidirectional TCN
        learning_rate: Initial learning rate
        weight_decay: AdamW weight decay
        max_epochs: For cosine annealing scheduler
    """

    def __init__(
        self,
        input_size: int = 44,
        hidden_size: int = 256,
        num_layers: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = True,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4,
        max_epochs: int = 30,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if bidirectional:
            self.model = HeartSoundSegmenterBiTCN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                kernel_size=kernel_size,
                dropout=dropout,
            )
        else:
            self.model = HeartSoundSegmenterTCN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                kernel_size=kernel_size,
                dropout=dropout,
            )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        num_classes = 4

        # Metrics
        self.train_metrics = MetricCollection(
            {
                "accuracy": Accuracy(task="multiclass", average="macro", num_classes=num_classes),
                "f1": F1Score(task="multiclass", average="macro", num_classes=num_classes),
                "precision": Precision(task="multiclass", average="macro", num_classes=num_classes),
                "recall": Recall(task="multiclass", average="macro", num_classes=num_classes),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

        # Per-class metrics for test
        self.test_metrics_per_class = MetricCollection(
            {
                "accuracy": Accuracy(task="multiclass", average=None, num_classes=num_classes),
                "f1": F1Score(task="multiclass", average=None, num_classes=num_classes),
                "precision": Precision(task="multiclass", average=None, num_classes=num_classes),
                "recall": Recall(task="multiclass", average=None, num_classes=num_classes),
            },
            prefix="test_",
        )

        # AUROC from CRF forward-backward marginals (updated on CPU; see test_step note)
        self.test_auroc = AUROC(task="multiclass", average="macro", num_classes=num_classes)
        self.test_auroc_per_class = AUROC(task="multiclass", average=None, num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass returning emission scores."""
        return self.model(x)

    def _compute_logits(self, x: Tensor) -> Tensor:
        """Get logits in (batch, classes, seq_len) format for metrics."""
        emissions = self.model(x)  # (B, T, K)
        return emissions.permute(0, 2, 1)  # (B, K, T)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        loss = self.model.loss(x, y)

        # Use emissions for metrics (faster than Viterbi decoding)
        logits = self._compute_logits(x)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log_dict(self.train_metrics(logits, y), prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.train_metrics.reset()

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        loss = self.model.loss(x, y)

        logits = self._compute_logits(x)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log_dict(self.val_metrics(logits, y), prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        self.val_metrics.reset()

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        loss = self.model.loss(x, y)

        # Use Viterbi decoding for test metrics (more accurate)
        predictions = self.model.decode(x)  # (B, T)

        # Convert to one-hot logits for metrics
        logits = torch.zeros(x.shape[0], 4, x.shape[1], device=x.device)
        logits.scatter_(1, predictions.unsqueeze(1), 10.0)

        # AUROC from CRF forward-backward marginals. Compute on CPU: torchmetrics AUROC returns
        # garbage on Apple MPS for the trained model's sharply-peaked probabilities.
        marginal_logits = self.model.marginals(x).permute(0, 2, 1).detach().cpu()  # (B, K, T)
        y_cpu = y.cpu()
        self.test_auroc.update(marginal_logits, y_cpu)
        self.test_auroc_per_class.update(marginal_logits, y_cpu)

        self.log("test_loss", loss)
        self.log_dict(self.test_metrics(logits, y))

        # Per-class metrics
        metrics_per_class = self.test_metrics_per_class(logits, y)
        for metric_name, metric_values in metrics_per_class.items():
            for i, v in enumerate(metric_values):
                self.log(f"{metric_name}_{i}", v)

        return loss

    def on_test_epoch_end(self) -> None:
        self.log("test_AUROC", self.test_auroc.compute())
        for i, v in enumerate(self.test_auroc_per_class.compute()):
            self.log(f"test_AUROC_{i}", v)

        self.test_metrics.reset()
        self.test_metrics_per_class.reset()
        self.test_auroc.reset()
        self.test_auroc_per_class.reset()

        # Print model info. RF is in working-resolution frames (seconds depend on the downsample
        # factor, which this module doesn't know), so report frames only.
        if hasattr(self.model, "bitcn"):
            rf = self.model.bitcn.receptive_field
            print("\nBidirectional TCN + CRF")
        else:
            rf = self.model.tcn.receptive_field
            print("\nUnidirectional TCN + CRF")
        print(f"Receptive Field: {rf} frames")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

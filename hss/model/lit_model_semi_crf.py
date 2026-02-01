"""PyTorch Lightning module for heart sound segmentation with Semi-Markov CRF."""

from typing import Tuple

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import MetricCollection
from torchmetrics.classification import AUROC, Accuracy, F1Score, Precision, Recall

from hss.model.segmenter_semi_crf import HeartSoundSegmenterSemiCRF


class LitModelSemiCRF(pl.LightningModule):
    """Lightning module using Semi-Markov CRF-based segmenter with duration modeling.

    This model extends the CRF approach by explicitly modeling segment durations
    using learnable Gaussian distributions per state, similar to Springer's HSMM.

    Args:
        input_size: Size of input features
        batch_size: Batch size for training
        device: Device to use
        max_duration: Maximum segment duration in frames
        duration_means: Initial mean duration for each state (S1, Systole, S2, Diastole)
        duration_stds: Initial std duration for each state
    """

    def __init__(
        self,
        input_size: int,
        batch_size: int,
        device: torch.device,
        max_duration: int = 500,
        duration_means: list[float] | None = None,
        duration_stds: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["device"])

        self.model = HeartSoundSegmenterSemiCRF(
            input_size=input_size,
            batch_size=batch_size,
            device=device,
            max_duration=max_duration,
            duration_means=duration_means,
            duration_stds=duration_stds,
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

        # AUROC uses marginal probabilities (from forward-backward) instead of decoded predictions
        self.test_auroc_per_class = AUROC(task="multiclass", average=None, num_classes=num_classes)
        self.test_auroc = AUROC(task="multiclass", average="macro", num_classes=num_classes)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _decode_to_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Decode Semi-Markov CRF and convert to one-hot logits for metrics."""
        decoded = self.model.decode(x)  # (batch_size, seq_len)
        batch_size, seq_len = decoded.shape

        # Create one-hot logits from decoded sequences (vectorized)
        logits = torch.zeros(batch_size, 4, seq_len, device=x.device)
        logits.scatter_(1, decoded.unsqueeze(1), 10.0)

        return logits

    def _marginals_to_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute marginal probabilities and format for metrics.

        Uses forward-backward algorithm to get P(y_t = k | x), which properly
        incorporates learned Semi-Markov CRF transition and duration constraints.
        """
        marginals = self.model.marginals(x)  # (batch_size, seq_len, num_tags)
        # Permute to (batch_size, num_tags, seq_len) for torchmetrics
        return marginals.permute(0, 2, 1)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        loss = self.model.loss(x, y)

        # Use raw emission logits for training metrics (faster than decoding)
        # Emissions are (B, T, K), need (B, K, T) for torchmetrics
        logits = self.model(x).permute(0, 2, 1)

        metrics_per_class = self.train_metrics_per_class(logits, y)
        self.train_metrics_per_class.reset()

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log_dict(self.train_metrics(logits, y), prog_bar=True, on_step=True, on_epoch=True)

        for metric_name, metric_values in metrics_per_class.items():
            for i, v in enumerate(metric_values):
                self.log(f"{metric_name}_{i}", v)

        # Log learned duration parameters
        if batch_idx == 0:
            dur_params = self.model.get_duration_params()
            for i, (mean, std) in enumerate(zip(dur_params["means"], dur_params["stds"], strict=True)):
                self.log(f"duration_mean_{i}", mean)
                self.log(f"duration_std_{i}", std)

        return loss

    def on_train_epoch_end(self) -> None:
        self.train_metrics_per_class.reset()
        self.train_metrics.reset()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch

        # Semi-Markov CRF loss
        loss = self.model.loss(x, y)

        # Use raw emission logits for validation metrics (faster than decoding)
        logits = self.model(x).permute(0, 2, 1)

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

        # Semi-Markov CRF loss
        loss = self.model.loss(x, y)

        # Decode for accuracy/precision/recall/F1 metrics
        decoded_logits = self._decode_to_logits(x)

        # Marginals for AUROC (forward-backward algorithm)
        marginal_logits = self._marginals_to_logits(x)

        metrics_per_class = self.test_metrics_per_class(decoded_logits, y)
        self.test_metrics_per_class.reset()

        # Update AUROC with marginal probabilities
        self.test_auroc_per_class.update(marginal_logits, y)
        self.test_auroc.update(marginal_logits, y)

        self.log("test_loss", loss)
        self.log_dict(self.test_metrics(decoded_logits, y))

        for metric_name, metric_values in metrics_per_class.items():
            for i, v in enumerate(metric_values):
                self.log(f"{metric_name}_{i}", v)

        return loss

    def on_test_epoch_end(self) -> None:
        # Log AUROC computed from marginal probabilities
        auroc_per_class = self.test_auroc_per_class.compute()
        for i, v in enumerate(auroc_per_class):
            self.log(f"test_MulticlassAUROC_{i}", v)
        self.log("test_MulticlassAUROC", self.test_auroc.compute())

        self.test_metrics_per_class.reset()
        self.test_metrics.reset()
        self.test_auroc_per_class.reset()
        self.test_auroc.reset()

        # Print final learned duration parameters
        dur_params = self.model.get_duration_params()
        state_names = ["S1", "Systole", "S2", "Diastole"]
        print("\nLearned Duration Parameters:")
        for i, name in enumerate(state_names):
            print(f"  {name}: μ={dur_params['means'][i]:.1f}, σ={dur_params['stds'][i]:.1f} frames")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = Adam(self.parameters(), lr=0.01)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9**epoch)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

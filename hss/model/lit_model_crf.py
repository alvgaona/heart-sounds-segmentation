"""PyTorch Lightning module for heart sound segmentation with CRF."""

from typing import Tuple

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import MetricCollection
from torchmetrics.classification import AUROC, Accuracy, F1Score, Precision, Recall

from hss.model.boundary_loss import BoundaryLossConfig, boundary_weighted_ce
from hss.model.segmenter_crf import HeartSoundSegmenterCRF
from hss.model.segmenter_xlstm import HeartSoundSegmenterXLSTMCRF


class LitModelCRF(pl.LightningModule):
    """Lightning module using CRF-based segmenter."""

    def __init__(
        self,
        input_size: int,
        batch_size: int,
        device: torch.device,
        lr: float = 0.01,
        boundary_cfg: BoundaryLossConfig | None = None,
        arch: str = "bilstm",
        hidden_size: int = 240,
        num_heads: int = 4,
        num_layers: int = 2,
        bidirectional: bool = True,
        phase: bool = False,
        multirate: bool = False,
    ) -> None:
        super().__init__()
        if arch == "xlstm":
            self.model = HeartSoundSegmenterXLSTMCRF(
                input_size=input_size,
                batch_size=batch_size,
                device=device,
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_layers=num_layers,
                bidirectional=bidirectional,
                phase=phase,
                multirate=multirate,
            )
        elif arch == "bilstm":
            self.model = HeartSoundSegmenterCRF(
                input_size=input_size,
                batch_size=batch_size,
                device=device,
                hidden_size=hidden_size,
                multirate=multirate,
                num_layers=num_layers,
            )
        else:
            raise ValueError(f"unknown arch {arch!r} (expected 'bilstm' or 'xlstm')")
        self.arch = arch
        self.batch_size = batch_size
        self.lr = lr
        self.boundary_cfg = boundary_cfg or BoundaryLossConfig()
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
        """Decode CRF and convert to one-hot logits for metrics."""
        decoded = self.model.decode(x)  # (batch_size, seq_len)
        batch_size, seq_len = decoded.shape

        # Create one-hot logits from decoded sequences (vectorized)
        logits = torch.zeros(batch_size, 4, seq_len, device=x.device)
        logits.scatter_(1, decoded.unsqueeze(1), 10.0)

        return logits

    def _marginals_to_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute marginal probabilities and format for metrics.

        Uses forward-backward algorithm to get P(y_t = k | x), which properly
        incorporates learned CRF transition constraints into the probabilities.
        """
        marginals = self.model.marginals(x)  # (batch_size, seq_len, num_tags)
        # Permute to (batch_size, num_tags, seq_len) for torchmetrics
        return marginals.permute(0, 2, 1)

    def _compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return (total loss, extras). Without the boundary aux this is the plain CRF NLL.

        With it, total = crf_nll + aux_lambda * boundary_weighted_ce, and emissions are computed once
        for both terms (avoids a second LSTM pass). Extras hold the detached nll/aux for logging.
        """
        if not self.boundary_cfg.enabled:
            return self.model.loss(x, y), {}
        emissions = self.model(x)
        nll = self.model.crf(emissions, y)
        aux = boundary_weighted_ce(emissions, y, self.boundary_cfg)
        total = nll + self.boundary_cfg.aux_lambda * aux
        return total, {"nll": nll.detach(), "aux": aux.detach()}

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        loss, extras = self._compute_loss(x, y)

        # Decode for metrics
        logits = self._decode_to_logits(x)

        metrics_per_class = self.train_metrics_per_class(logits, y)
        self.train_metrics_per_class.reset()

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        for name, value in extras.items():
            self.log(f"train_{name}", value, on_step=True, on_epoch=True)
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

        # CRF loss (plus boundary aux when enabled)
        loss, extras = self._compute_loss(x, y)

        # Decode for metrics
        logits = self._decode_to_logits(x)

        metrics_per_class = self.val_metrics_per_class(logits, y)
        self.val_metrics_per_class.reset()

        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        for name, value in extras.items():
            self.log(f"val_{name}", value, on_step=False, on_epoch=True)
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

        # CRF loss (plus boundary aux when enabled)
        loss, _ = self._compute_loss(x, y)

        # Decode for accuracy/precision/recall/F1 metrics
        decoded_logits = self._decode_to_logits(x)

        # Marginals for AUROC (forward-backward algorithm). Compute AUROC on CPU: torchmetrics'
        # AUROC sort/threshold logic returns garbage on Apple MPS for the trained model's sharply
        # peaked probabilities (observed 0.0 / huge values on MPS vs a correct ~0.99 on CPU).
        marginal_logits = self._marginals_to_logits(x).detach().cpu()
        y_cpu = y.cpu()

        metrics_per_class = self.test_metrics_per_class(decoded_logits, y)
        self.test_metrics_per_class.reset()

        # Update AUROC with marginal probabilities (on CPU, see note above)
        self.test_auroc_per_class.update(marginal_logits, y_cpu)
        self.test_auroc.update(marginal_logits, y_cpu)

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
            self.log(f"test_AUROC_{i}", v)
        self.log("test_AUROC", self.test_auroc.compute())

        self.test_metrics_per_class.reset()
        self.test_metrics.reset()
        self.test_auroc_per_class.reset()
        self.test_auroc.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9**epoch)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

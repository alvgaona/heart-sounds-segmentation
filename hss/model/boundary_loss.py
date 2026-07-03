"""Boundary-aware auxiliary loss for CRF segmentation.

Adds a per-frame weighted cross-entropy on the encoder emissions to the CRF NLL. The weight map
emphasizes (a) frames within a window of a ground-truth state transition (boundary sharpening) and
(b) chosen classes (S1 by default). This targets the observed failure mode — missed/spurious S1
detections (cycle miscounts) — via recall on S1 and crisper onsets, rather than boundary jitter.

The aux CE operates on the raw emissions (LSTM linear output), which are exactly what the linear-chain
CRF consumes; since that CRF is decoder-invariant, sharpening emissions is the effective lever.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BoundaryLossConfig:
    """Configuration for the boundary-aware auxiliary loss.

    Attributes:
        aux_lambda: Weight of the aux CE relative to the CRF NLL. 0 disables the aux loss entirely.
        boundary_weight: Multiplier applied to frames within the boundary window (1.0 = no boundary
            emphasis).
        boundary_window: Half-width (in frames) of the boundary emphasis region around each ground-truth
            transition. 0 disables boundary emphasis.
        class_weights: Per-class multiplier applied to every frame by its label (index 0 = S1).
    """

    aux_lambda: float = 0.0
    boundary_weight: float = 2.0
    boundary_window: int = 2
    class_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)

    @property
    def enabled(self) -> bool:
        return self.aux_lambda > 0.0


def boundary_weight_map(tags: torch.Tensor, cfg: BoundaryLossConfig) -> torch.Tensor:
    """Per-frame loss weights (boundary emphasis multiplied by class emphasis).

    Args:
        tags: Ground-truth labels of shape (batch_size, seq_len).
        cfg: Boundary-loss configuration.

    Returns:
        Weight tensor of shape (batch_size, seq_len), dtype float32.
    """
    weights = torch.ones(tags.shape, device=tags.device, dtype=torch.float32)

    if cfg.boundary_weight != 1.0 and cfg.boundary_window > 0:
        transition = torch.zeros(tags.shape, device=tags.device, dtype=torch.float32)
        transition[:, 1:] = (tags[:, 1:] != tags[:, :-1]).float()
        w = cfg.boundary_window
        dilated = F.max_pool1d(transition.unsqueeze(1), kernel_size=2 * w + 1, stride=1, padding=w).squeeze(1)
        weights = torch.where(dilated > 0, weights * cfg.boundary_weight, weights)

    class_weights = torch.tensor(cfg.class_weights, device=tags.device, dtype=torch.float32)
    return weights * class_weights[tags]


def boundary_weighted_ce(emissions: torch.Tensor, tags: torch.Tensor, cfg: BoundaryLossConfig) -> torch.Tensor:
    """Weighted per-frame cross-entropy on emissions (weighted mean over all frames).

    Args:
        emissions: Emission scores of shape (batch_size, seq_len, num_tags).
        tags: Ground-truth labels of shape (batch_size, seq_len).
        cfg: Boundary-loss configuration.

    Returns:
        Scalar weighted-mean cross-entropy.
    """
    batch_size, seq_len, num_tags = emissions.shape
    log_probs = F.log_softmax(emissions, dim=-1)
    ce = F.nll_loss(log_probs.reshape(-1, num_tags), tags.reshape(-1), reduction="none").reshape(batch_size, seq_len)
    weights = boundary_weight_map(tags, cfg)
    return (ce * weights).sum() / weights.sum().clamp_min(1.0)

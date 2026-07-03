"""Tests for the boundary-aware auxiliary loss."""

import torch
import torch.nn.functional as F

from hss.model.boundary_loss import BoundaryLossConfig, boundary_weight_map, boundary_weighted_ce


TAGS = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 3]])  # transitions at indices 2, 5, 7


def test_enabled_property() -> None:
    assert not BoundaryLossConfig(aux_lambda=0.0).enabled
    assert BoundaryLossConfig(aux_lambda=0.5).enabled


def test_weight_map_uniform_when_no_emphasis() -> None:
    cfg = BoundaryLossConfig(aux_lambda=0.5, boundary_weight=1.0, class_weights=(1.0, 1.0, 1.0, 1.0))
    weights = boundary_weight_map(TAGS, cfg)
    assert torch.allclose(weights, torch.ones_like(weights))


def test_weight_map_boundary_dilation() -> None:
    cfg = BoundaryLossConfig(aux_lambda=0.5, boundary_weight=2.0, boundary_window=1, class_weights=(1.0, 1.0, 1.0, 1.0))
    weights = boundary_weight_map(TAGS, cfg)
    # frame 0 is >window from the first transition (idx 2); every other frame is within +/-1 of a transition
    expected = torch.tensor([[1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]])
    assert torch.allclose(weights, expected)


def test_weight_map_class_emphasis() -> None:
    cfg = BoundaryLossConfig(aux_lambda=0.5, boundary_weight=1.0, class_weights=(3.0, 1.0, 1.0, 1.0))
    weights = boundary_weight_map(TAGS, cfg)
    # only the two S1 frames (indices 0, 1) get the 3x class weight
    expected = torch.tensor([[3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    assert torch.allclose(weights, expected)


def test_weight_map_boundary_and_class_combine_multiplicatively() -> None:
    cfg = BoundaryLossConfig(aux_lambda=0.5, boundary_weight=2.0, boundary_window=1, class_weights=(3.0, 1.0, 1.0, 1.0))
    weights = boundary_weight_map(TAGS, cfg)
    # frame 0: S1, not boundary -> 3; frame 1: S1 and boundary -> 6; rest: boundary only -> 2
    expected = torch.tensor([[3.0, 6.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]])
    assert torch.allclose(weights, expected)


def test_weighted_ce_reduces_to_mean_ce_when_uniform() -> None:
    torch.manual_seed(0)
    emissions = torch.randn(2, 8, 4)
    tags = TAGS.repeat(2, 1)
    cfg = BoundaryLossConfig(aux_lambda=0.5, boundary_weight=1.0, class_weights=(1.0, 1.0, 1.0, 1.0))
    got = boundary_weighted_ce(emissions, tags, cfg)
    expected = F.cross_entropy(emissions.reshape(-1, 4), tags.reshape(-1))
    assert torch.allclose(got, expected)


def test_weighted_ce_gradients_flow() -> None:
    emissions = torch.randn(2, 8, 4, requires_grad=True)
    tags = TAGS.repeat(2, 1)
    cfg = BoundaryLossConfig(aux_lambda=0.5, boundary_weight=2.0, boundary_window=2, class_weights=(2.0, 1.0, 1.0, 1.0))
    loss = boundary_weighted_ce(emissions, tags, cfg)
    loss.backward()
    assert emissions.grad is not None
    assert torch.isfinite(loss)

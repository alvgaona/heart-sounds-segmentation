"""Tests for the fixed-lag streaming constrained Viterbi (Experiment B, phase 2)."""

import torch

from hss.utils.streaming_decode import (
    constrained_viterbi,
    stream_decode,
    valid_transition_fraction,
)


def _clean_cycle_emissions(cycles: int = 4, noise: float = 0.5) -> tuple[torch.Tensor, list[int]]:
    """Emissions that strongly favor a repeating S1->Sys->S2->Dia cycle, plus a true-label list."""
    pattern = [0] * 4 + [1] * 8 + [2] * 4 + [3] * 12  # one cardiac cycle at 50 Hz-ish proportions
    labels = pattern * cycles
    t = len(labels)
    torch.manual_seed(0)
    emis = noise * torch.randn(t, 4, dtype=torch.float64)
    emis[range(t), labels] += 5.0  # strong evidence for the true state
    return emis, labels


def test_constrained_viterbi_is_valid():
    emis, _ = _clean_cycle_emissions()
    path = constrained_viterbi(emis)
    assert valid_transition_fraction(path) == 1.0


def test_constrained_viterbi_recovers_clean_cycle():
    emis, labels = _clean_cycle_emissions()
    path = constrained_viterbi(emis).tolist()
    agree = sum(a == b for a, b in zip(path, labels, strict=True)) / len(labels)
    assert agree > 0.95


def test_stream_full_lag_equals_offline():
    emis, _ = _clean_cycle_emissions()
    t = emis.shape[0]
    offline = constrained_viterbi(emis)
    streamed = stream_decode(emis, lag=t)  # lag >= T-1 must reproduce offline exactly
    assert torch.equal(offline, streamed)


def test_stream_bounded_lag_stays_valid_on_clean_signal():
    emis, _ = _clean_cycle_emissions()
    for lag in (5, 10, 20):
        path = stream_decode(emis, lag=lag)
        assert valid_transition_fraction(path) == 1.0, f"lag={lag} produced an invalid transition"


def test_valid_transition_fraction_detects_invalid():
    assert valid_transition_fraction(torch.tensor([0, 1, 2, 3, 0])) == 1.0  # valid cycle
    assert valid_transition_fraction(torch.tensor([0, 2])) == 0.0  # S1 -> S2 skips systole

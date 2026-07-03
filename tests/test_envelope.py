"""Tests for the amplitude envelope features."""

import numpy as np

from hss.transforms.envelope import envelope_features, hilbert_envelope, homomorphic_envelope


FS = 1000


def _burst_signal() -> np.ndarray:
    """1 s of silence-then-burst: a 100 Hz tone in the middle third only."""
    t = np.arange(FS) / FS
    tone = np.sin(2 * np.pi * 100 * t)
    mask = (t > 0.33) & (t < 0.66)
    return (tone * mask).astype(np.float64)


def test_hilbert_envelope_tracks_amplitude() -> None:
    # amplitude-modulated tone: envelope should recover the (positive) modulation
    t = np.arange(FS) / FS
    mod = 1.0 + 0.5 * np.sin(2 * np.pi * 2 * t)
    x = mod * np.sin(2 * np.pi * 100 * t)
    env = hilbert_envelope(x)
    assert env.shape == x.shape
    assert np.corrcoef(env, mod)[0, 1] > 0.8


def test_homomorphic_envelope_higher_on_burst() -> None:
    x = _burst_signal()
    env = homomorphic_envelope(x, FS)
    mid = env[int(0.4 * FS) : int(0.6 * FS)].mean()
    edge = np.concatenate([env[: int(0.2 * FS)], env[int(0.8 * FS) :]]).mean()
    assert mid > edge


def test_envelope_features_shape_and_norm() -> None:
    feats = envelope_features(_burst_signal(), FS)
    assert feats.shape == (FS, 2)
    assert np.isfinite(feats).all()
    assert np.allclose(feats.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(feats.std(axis=0), 1.0, atol=1e-2)


def test_envelope_features_accepts_2d_column() -> None:
    x = _burst_signal().reshape(-1, 1)
    feats = envelope_features(x, FS)
    assert feats.shape == (FS, 2)

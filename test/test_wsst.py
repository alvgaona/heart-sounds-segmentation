import numpy as np
import pytest
import scipy
import torch

from hss.transforms import FSST, WSST


@pytest.fixture
def signal() -> torch.Tensor:
    """A deterministic 2 s frame at 1000 Hz (S1/S2-like bursts on noise)."""
    rng = np.random.default_rng(0)
    t = np.arange(2000) / 1000.0
    x = 0.1 * rng.standard_normal(2000)
    for center in (0.2, 0.7, 1.2, 1.7):
        x += np.exp(-((t - center) ** 2) / (2 * 0.01**2)) * np.sin(2 * np.pi * 50 * t)
    return torch.tensor(x, dtype=torch.float32)


def test_wsst_stack_shape_and_norm(signal: torch.Tensor) -> None:
    out = WSST(1000, wavelet="amor", num_voices=8, truncate_freq=(25, 200), stack=True)(signal)
    assert out.shape == (2000, 48)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    half = out.shape[1] // 2
    for part in (out[:, :half], out[:, half:]):
        assert part.mean().abs() < 1e-4
        assert abs(part.std().item() - 1.0) < 1e-3


def test_wsst_abs_mode(signal: torch.Tensor) -> None:
    out = WSST(1000, num_voices=8, truncate_freq=(25, 200), abs=True)(signal)
    assert out.shape == (2000, 24)
    assert out.dtype == torch.float32
    assert (out >= 0).all()


def test_wsst_num_voices_scales_dimension(signal: torch.Tensor) -> None:
    dims = {nv: WSST(1000, num_voices=nv, truncate_freq=(25, 200), stack=True)(signal).shape[1] for nv in (8, 16)}
    assert dims[16] > dims[8]


def test_wsst_deterministic(signal: torch.Tensor) -> None:
    wsst = WSST(1000, num_voices=8, truncate_freq=(25, 200), stack=True)
    assert torch.equal(wsst(signal), wsst(signal))


def test_fsst_unchanged_by_refactor(signal: torch.Tensor) -> None:
    window = scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=False)
    fsst = FSST(1000, window=window, truncate_freq=(25, 200), stack=True)
    out = fsst(signal)
    assert out.shape == (2000, 44)
    assert torch.equal(out, fsst(signal))

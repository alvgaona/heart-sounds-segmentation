"""PCG envelope features (Springer 2016 style) — for fusion with FSST or as a standalone feature set.

Springer's HSMM used four features; `springer_features` returns all four, `envelope_features` the two
amplitude envelopes used for FSST fusion:

  homomorphic_envelope: exp(LPF_8Hz(log|hilbert(x)|))  -- smooth amplitude envelope
  hilbert_envelope:     |hilbert(x)|                    -- instantaneous amplitude
  wavelet_envelope:     |hilbert(bandpass 62.5-125 Hz)| -- band-energy proxy for Springer's rbio3.9 L3
                                                           detail (pywt not installed; band matches L3)
  psd_envelope:         40-60 Hz spectrogram band power over time

All are computed on a band-passed (25-400 Hz) copy of the signal, matching Springer's front-end.
"""

from typing import cast

import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def _butter(order: int, wn: float | list[float], btype: str) -> tuple[np.ndarray, np.ndarray]:
    # cast: scipy's butter stub unions in `None`, but with output="ba" it always returns (b, a).
    return cast("tuple[np.ndarray, np.ndarray]", butter(order, wn, btype))


def bandpass(x: np.ndarray, fs: float, low: float = 25.0, high: float = 400.0, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth band-pass."""
    b, a = _butter(order, [2 * low / fs, 2 * high / fs], "band")
    return filtfilt(b, a, x)


def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    """Instantaneous amplitude (magnitude of the analytic signal)."""
    return np.abs(cast("np.ndarray", hilbert(x)))


def homomorphic_envelope(x: np.ndarray, fs: float, cutoff: float = 8.0) -> np.ndarray:
    """Smooth amplitude envelope via homomorphic filtering (1st-order LPF at `cutoff` Hz)."""
    b, a = _butter(1, 2 * cutoff / fs, "low")
    return np.exp(filtfilt(b, a, np.log(hilbert_envelope(x) + 1e-8)))


def wavelet_envelope(x: np.ndarray, fs: float, band: tuple[float, float] = (62.5, 125.0)) -> np.ndarray:
    """Band-energy envelope: |hilbert(bandpass(x, band))|.

    A dependency-free proxy for Springer's wavelet feature (|rbio3.9 level-3 detail|); the default band
    matches that level's pass-band at fs=1000 Hz. Swap in a pywt DWT for exact fidelity.
    """
    return hilbert_envelope(bandpass(x, fs, low=band[0], high=band[1]))


def psd_envelope(x: np.ndarray, fs: float, band: tuple[float, float] = (40.0, 60.0), nperseg: int = 128) -> np.ndarray:
    """Time-varying power in `band` via a short-time FFT, interpolated back to the signal length.

    Implemented with numpy FFT (not scipy.signal.spectrogram) to keep the module's scipy type stubs clean.
    """
    nperseg = min(nperseg, len(x))
    hop = max(1, nperseg // 2)
    win = np.hanning(nperseg)
    freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not mask.any():
        mask[:] = True
    starts = list(range(0, len(x) - nperseg + 1, hop)) or [0]
    centers = [s + nperseg / 2 for s in starts]
    power = [np.abs(np.fft.rfft(x[s : s + nperseg] * win)) ** 2 for s in starts]
    band_power = [p[mask].mean() for p in power]
    return np.interp(np.arange(len(x)), centers, band_power)


def _znorm(feats: np.ndarray) -> np.ndarray:
    return ((feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-8)).astype(np.float32)


def envelope_features(signal: np.ndarray, fs: float) -> np.ndarray:
    """Return (T, 2) z-normalised [homomorphic, Hilbert] envelopes of the band-passed signal.

    Each channel is standardised over time (zero mean, unit variance) so the two envelopes share a
    scale; the caller can rescale to match the FSST magnitude before concatenation.
    """
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    bp = bandpass(x, fs)
    return _znorm(np.stack([homomorphic_envelope(bp, fs), hilbert_envelope(bp)], axis=-1))


def springer_features(signal: np.ndarray, fs: float) -> np.ndarray:
    """Return (T, 4) z-normalised Springer feature set [homomorphic, Hilbert, wavelet, PSD].

    This is the standalone Springer 2016 feature set (no FSST), for comparing hand-crafted features
    against the FSST front-end under the same model and CV protocol.
    """
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    bp = bandpass(x, fs)
    feats = np.stack(
        [homomorphic_envelope(bp, fs), hilbert_envelope(bp), wavelet_envelope(x, fs), psd_envelope(x, fs)],
        axis=-1,
    )
    return _znorm(feats)

"""Amplitude envelope features for PCG, for fusion with FSST (Springer 2016 style).

The FSST (25-200 Hz synchrosqueezed) does not always surface faint S1 sounds; the amplitude envelope
does (see the Pass 2 analysis). These two envelopes, concatenated onto the FSST, give the model a
direct amplitude cue for S1 detection.

  homomorphic_envelope: exp(LPF_8Hz(log|hilbert(x)|))  -- smooth amplitude envelope
  hilbert_envelope:     |hilbert(x)|                    -- instantaneous amplitude

Both are computed on a band-passed (25-400 Hz) copy of the signal, matching Springer's front-end.
"""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def bandpass(x: np.ndarray, fs: float, low: float = 25.0, high: float = 400.0, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth band-pass."""
    b, a = butter(order, [2 * low / fs, 2 * high / fs], "band")
    return filtfilt(b, a, x)


def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    """Instantaneous amplitude (magnitude of the analytic signal)."""
    return np.abs(hilbert(x))


def homomorphic_envelope(x: np.ndarray, fs: float, cutoff: float = 8.0) -> np.ndarray:
    """Smooth amplitude envelope via homomorphic filtering (1st-order LPF at `cutoff` Hz)."""
    b, a = butter(1, 2 * cutoff / fs, "low")
    return np.exp(filtfilt(b, a, np.log(hilbert_envelope(x) + 1e-8)))


def envelope_features(signal: np.ndarray, fs: float) -> np.ndarray:
    """Return (T, 2) z-normalised [homomorphic, Hilbert] envelopes of the band-passed signal.

    Each channel is standardised over time (zero mean, unit variance) so the two envelopes share a
    scale; the caller can rescale to match the FSST magnitude before concatenation.
    """
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    bp = bandpass(x, fs)
    feats = np.stack([homomorphic_envelope(bp, fs), hilbert_envelope(bp)], axis=-1)  # (T, 2)
    feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-8)
    return feats.astype(np.float32)

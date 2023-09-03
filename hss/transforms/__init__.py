from typing import Optional, Tuple

import numpy
import numpy as np
import scipy
import ssqueezepy as ssq
import torch


class Resample:
    def __init__(self, num: int) -> None:
        """
        Args:
            num (int): number of output samples
        """
        self.num = num

    def __call__(self, x) -> np.ndarray:
        """
        Args:
            x: input signal to get resampled

        Returns:
            (np.ndarray): resampled signal
        """
        return scipy.signal.resample(x, self.num)


class FSST:
    """
    Fourier Synchrosqueezed Transform
    """

    def __init__(self, fs: float, flipud: bool = False, window: Optional[numpy.ndarray] = None):
        """
        Args:
            fs (float): sample frequency
            flipud (bool): ?. Default: False
            window (numpy.ndarray): window provided to compute the transform. Default: None
        """
        self.flipud = flipud
        self.fs = fs
        self.window = window

    def __call__(self, x: torch.Tensor) -> Tuple:
        """
        Computes the transform

        Args:
            x (torch.Tensor): input signal

        Returns:
            Tuple: Original signal, STFT Synchrosqueezed Transform, STFT, and frequencies

        """
        fsst, stft, f, *_ = ssq.ssq_stft(
            x,
            flipud=self.flipud,
            fs=self.fs,
            window=self.window,
        )

        return x, fsst, stft, np.ascontiguousarray(f)

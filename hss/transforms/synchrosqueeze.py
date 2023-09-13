from typing import Optional, Tuple

import numpy as np
import ssqueezepy as ssq
import torch


class FSST:
    """
    Fourier Synchrosqueezed Transform
    """

    def __init__(
        self, fs: float, flipud: bool = False, window: Optional[np.ndarray] = None
    ):
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

        if isinstance(f, torch.Tensor) and f.get_device() > -1:
            f = f.cpu()

        return x, fsst, stft, np.ascontiguousarray(f)

from typing import  Optional, Tuple

import numpy as np
import ssqueezepy as ssq
import torch


class FSST:
    """
    Fourier Synchrosqueezed Transform
    """

    def __init__(
        self,
        fs: float,
        flipud: bool = False,
        window: Optional[np.ndarray] = None,
        stack: bool = False,
        truncate_freq: Optional[tuple] = None,
    ):
        """
        Args:
            fs (float): sample frequency
            flipud (bool): ?. Default: False
            window (numpy.ndarray): window provided to compute the transform. Default: None
            stack (bool): true or false in order to stack or not the real and image parts of the spectrum. Default: False
        """
        self.flipud = flipud
        self.fs = fs
        self.window = window
        self.stack = stack
        self.truncate_freq = truncate_freq

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the transform

        Args:
            x (torch.Tensor): input signal

        Returns:
            Tuple: Original signal, STFT Synchrosqueezed Transform, STFT, and frequencies

        """
        fsst, _, f, *_ = ssq.ssq_stft(
            x.cpu(),
            flipud=self.flipud,
            fs=self.fs,
            window=self.window,
        )

        if self.truncate_freq:
            f = self._truncate_frequencies(f.contiguous())

        if self.stack:
            return self._stack_real_imag(fsst, f)

        return fsst

    @classmethod
    def _stack_real_imag(cls, s: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Stack real and image part of the transform.

        Args:
            s (torch.Tensor): a time-frequency spectrum.

        Return:
            (torch.Tensor): the output will be the real and image values stacked on each other for each frequency.
        """
        s = torch.t(s)

        z = torch.zeros((s.shape[0], f.shape[0]), dtype=torch.complex64)

        for j in range(len(f)):
            z[:, j] = s[:, j]

        return torch.hstack((z.real, z.imag))

    def _truncate_frequencies(self, f: torch.Tensor) -> torch.Tensor:
        f = f.squeeze(0)
        indices = torch.logical_and(f >= self.truncate_freq[0], f <= self.truncate_freq[1])
        return f[indices]

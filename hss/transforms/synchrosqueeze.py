from typing import Optional

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
        abs: bool = False,
        stack: bool = False,
        truncate_freq: Optional[tuple] = None,
    ):
        """
        Args:
            fs (float): sample frequency
            flipud (bool): ?. Default: False
            window (numpy.ndarray): window provided to compute the transform. Default: None
            stack (bool): true or false in order to stack or not the real and image parts of the spectrum.
                Default: False
        """
        self.flipud = flipud
        self.fs = fs
        self.window = window
        self.abs = abs
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

        if isinstance(fsst, np.ndarray):
            fsst = torch.tensor(fsst)

        if isinstance(f, np.ndarray):
            f = torch.tensor(f)

        if self.truncate_freq:
            fsst, f = self._truncate_frequencies(fsst, f.contiguous())

        if self.abs:
            return torch.abs(fsst).t()

        if self.stack:
            return self._stack_real_imag(fsst)

        return fsst

    def _stack_real_imag(self, s: torch.Tensor) -> torch.Tensor:
        """
        Stack real and image part of the transform.

        Args:
            s (torch.Tensor): a time-frequency spectrum.

        Return:
            (torch.Tensor): the output will be the real and image values stacked on each other for each frequency.
        """
        s = torch.t(s)

        z = torch.zeros((s.shape[0], s.shape[1] * (2 if self.stack else 1)))

        for j in range(0, s.shape[1], 2):
            r = s[:, j].real
            i = s[:, j].imag

            z[:, j] = (r - torch.mean(r)) / torch.std(r, unbiased=True)
            z[:, j + 1] = (i - torch.mean(i)) / torch.std(i, unbiased=True)

        return z

    def _truncate_frequencies(self, s: torch.Tensor, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = f.squeeze(0)

        if self.truncate_freq:
            indices = torch.logical_and(f >= self.truncate_freq[0], f <= self.truncate_freq[1])
            return s[indices, :], f[indices]

        raise ValueError(f"Truncate frequency is: {self.truncate_freq}")

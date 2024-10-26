from typing import Optional

import numpy as np
import torch
from fsst import fftsqueeze


class FSST:
    """
    Fourier Synchrosqueezed Transform
    """

    def __init__(
        self,
        fs: float,
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
        s, f, t = fftsqueeze(x.cpu(), self.fs, self.window)
        s, f, t = torch.tensor(s), torch.tensor(f), torch.tensor(t)

        # TODO: Normalize FSST output

        if self.truncate_freq:
            s, f = self._truncate_frequencies(s, f.contiguous())

        if self.abs:
            return torch.abs(s).t()

        if self.stack:
            return self._stack_real_imag(s)

        return s

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
        """Truncate frequencies outside specified range.

        Args:
            s: Time-frequency spectrum tensor
            f: Frequency tensor

        Returns:
            Tuple of truncated spectrum and frequency tensors

        Raises:
            ValueError: If truncate_freq is not set
        """
        if not self.truncate_freq:
            raise ValueError(f"truncate_freq must be set, got: {self.truncate_freq}")

        f = f.squeeze(0)
        min_freq, max_freq = self.truncate_freq
        indices = torch.logical_and(f >= min_freq, f <= max_freq)

        return s[indices, :], f[indices]

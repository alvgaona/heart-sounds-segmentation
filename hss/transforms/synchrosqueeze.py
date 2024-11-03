from typing import Optional

import numpy.typing as npt
import torch
from fsst import fsst


class FSST:
    """
    Fourier Synchrosqueezed Transform
    """

    def __init__(
        self,
        fs: float,
        window: npt.NDArray,
        abs: bool = False,
        stack: bool = False,
        truncate_freq: Optional[tuple] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            fs (float): sample frequency
            flipud (bool): ?. Default: False
            window (numpy.ndarray): window provided to compute the transform. Default: None
            stack (bool): true or false in order to stack or not the real and image parts of the spectrum.
                Default: False
        """
        self.fs: float = fs
        self.window: npt.NDArray = window
        self.abs = abs
        self.stack = stack
        self.truncate_freq = truncate_freq
        self.dtype = dtype

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the transform

        Args:
            x (torch.Tensor): input signal

        Returns:
            Tuple: Original signal, STFT Synchrosqueezed Transform, STFT, and frequencies

        """
        s, f, t = fsst(x.numpy(), self.fs, self.window)

        s, f, t = (
            torch.tensor(s, dtype=torch.complex64),
            torch.tensor(f, dtype=self.dtype),
            torch.tensor(t, dtype=self.dtype),
        )

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
        # Calculate separate means and stds for real and imaginary parts
        real_mean = torch.mean(s.real)
        real_std = torch.std(s.real)
        imag_mean = torch.mean(s.imag)
        imag_std = torch.std(s.imag)

        # Normalize real and imaginary parts separately
        real_part = (s.real - real_mean) / real_std
        imag_part = (s.imag - imag_mean) / imag_std

        z = torch.cat([real_part, imag_part], dim=0)

        return z.t()

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

from typing import Optional

import numpy.typing as npt
import ssq
import torch


class _Synchrosqueeze:
    """
    Base for synchrosqueezed time-frequency transforms (FSST/WSST).

    Subclasses implement `_transform` to return the raw `(spectrum, frequencies, times)` numpy tuple from the
    `ssq` backend; the shared `__call__` handles frequency truncation, magnitude/real-imag stacking and dtype.
    """

    def __init__(
        self,
        fs: float,
        abs: bool = False,
        stack: bool = False,
        truncate_freq: Optional[tuple] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.fs: float = fs
        self.abs = abs
        self.stack = stack
        self.truncate_freq = truncate_freq
        self.dtype = dtype

    def _transform(self, x: torch.Tensor) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        raise NotImplementedError

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the transform.

        Args:
            x (torch.Tensor): input signal

        Returns:
            torch.Tensor: the (optionally truncated) spectrum, either raw complex, magnitude, or real/imag stacked.
        """
        s, f, t = self._transform(x)

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


class FSST(_Synchrosqueeze):
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
            window (numpy.ndarray): window provided to compute the transform.
            abs (bool): return the magnitude of the spectrum instead of the complex/stacked form. Default: False
            stack (bool): true or false in order to stack or not the real and image parts of the spectrum.
                Default: False
            truncate_freq (tuple): (min_freq, max_freq) band kept before abs/stack. Default: None
            dtype (torch.dtype): dtype for the real-valued frequency/time axes. Default: torch.float32
        """
        super().__init__(fs, abs=abs, stack=stack, truncate_freq=truncate_freq, dtype=dtype)
        self.window: npt.NDArray = window

    def _transform(self, x: torch.Tensor) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        return ssq.fsst(x.numpy(), self.fs, self.window)


class WSST(_Synchrosqueeze):
    """
    Wavelet Synchrosqueezed Transform
    """

    def __init__(
        self,
        fs: float,
        wavelet: str = "amor",
        num_voices: int = 32,
        abs: bool = False,
        stack: bool = False,
        truncate_freq: Optional[tuple] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            fs (float): sample frequency
            wavelet (str): mother wavelet, 'amor' (analytic Morlet) or 'bump'. Default: 'amor'
            num_voices (int): voices per octave; sets the number of log-spaced frequency bins. Default: 32
            abs (bool): return the magnitude of the spectrum instead of the complex/stacked form. Default: False
            stack (bool): true or false in order to stack or not the real and image parts of the spectrum.
                Default: False
            truncate_freq (tuple): (min_freq, max_freq) band kept before abs/stack. Default: None
            dtype (torch.dtype): dtype for the real-valued frequency/time axes. Default: torch.float32
        """
        super().__init__(fs, abs=abs, stack=stack, truncate_freq=truncate_freq, dtype=dtype)
        self.wavelet: str = wavelet
        self.num_voices: int = num_voices

    def _transform(self, x: torch.Tensor) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        return ssq.wsst(x.numpy(), self.fs, self.wavelet, self.num_voices)

import scipy
import torch


class Resample:
    def __init__(self, num: int) -> None:
        """
        Args:
            num (int): number of output samples
        """
        self.num = num

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input signal to get resampled

        Returns:
            (np.ndarray): resampled signal
        """
        return torch.tensor(scipy.signal.resample(x.cpu(), self.num), dtype=torch.float32)

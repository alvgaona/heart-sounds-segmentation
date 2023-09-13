import numpy as np
import scipy


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

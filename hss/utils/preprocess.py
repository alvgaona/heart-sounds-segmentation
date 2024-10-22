from math import floor
from typing import List, Tuple

import numpy as np
import torch


def frame_signal(
    x: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    stride: int,
    n: int,
) -> Tuple[List[torch.Tensor | np.ndarray], ...]:
    """
    Frame a given time-series input based on the length of the output frames and
    the desired stride.

    Can also frame labels associated with the input time-series data.

    Args:
        x (torch.Tensor | np.ndarray): the input data.
        y (torch.Tensor | np.ndarray): the labels associated with the input
        stride (float): the stride used to construct the frames.
        n (int): the number of samples or length the output frames will have.

    Returns:
        (dict[str, List[torch.Tensor | np.ndarray]]): the output in a dictionary format with
        the frames and labels.
    """
    if len(x.shape) == 1:
        x = x.unsqueeze(1)
    if len(y.shape) == 1:
        y = y.unsqueeze(1)

    T = x.shape[0]
    L = floor((T - n - 1) / stride)

    frames = []
    labels = []

    for i in range(L):
        frames.append(x[i * stride : i * stride + n, :])
        labels.append(y[i * stride : i * stride + n, :])

    if L <= 0:
        frames.append(x[:n, :])
        labels.append(y[:n, :])

    return frames, labels

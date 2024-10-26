from math import floor
from typing import List, Tuple

import torch


def frame_signal(
    x: torch.Tensor,
    y: torch.Tensor,
    stride: int,
    n: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Frame a given time-series input based on the length of the output frames and
    the desired stride.

    Can also frame labels associated with the input time-series data.

    Args:
        x (torch.Tensor): the input data
        y (torch.Tensor): the labels associated with the input
        stride (int): the stride used to construct the frames
        n (int): the number of samples or length the output frames will have

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor]]: A tuple containing:
            - List of framed input tensors
            - List of framed label tensors
    """
    # Ensure inputs are 2D
    if len(x.shape) == 1:
        x = x.unsqueeze(1)
    if len(y.shape) == 1:
        y = y.unsqueeze(1)

    # Validate input dimensions
    assert x.shape[0] == y.shape[0]

    T = x.shape[0]
    L = floor((T - n) / stride)

    frames = []
    labels = []

    frames = []
    labels = []

    for i in range(L):
        start_idx = i * stride
        end_idx = start_idx + n
        frames.append(x[start_idx:end_idx, :])
        labels.append(y[start_idx:end_idx, :])

    if L <= 0:
        frames.append(x[:n, :])
        labels.append(y[:n, :])

    return frames, labels

import pytest
import torch

from hss.transforms import Resample


@pytest.fixture
def resample_transform():
    return Resample(num=100)


@pytest.fixture
def input_tensor():
    return torch.tensor([1, 2, 3, 5])


def test_resample_output_type(resample_transform, input_tensor):
    y = resample_transform(input_tensor)
    assert isinstance(y, torch.Tensor)


def test_resample_output_shape(resample_transform, input_tensor):
    y = resample_transform(input_tensor)
    assert y.shape == (100,)


def test_resample_preserves_range(resample_transform, input_tensor):
    y = resample_transform(input_tensor)
    assert torch.min(y) >= torch.min(input_tensor)
    assert torch.max(y) <= torch.max(input_tensor)


def test_resample_different_input_sizes():
    f = Resample(num=50)
    x1 = torch.tensor([1, 2, 3])
    x2 = torch.tensor([1, 2, 3, 4, 5])

    y1 = f(x1)
    y2 = f(x2)

    assert y1.shape == (50,)
    assert y2.shape == (50,)

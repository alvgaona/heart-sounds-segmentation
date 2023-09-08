import numpy
import numpy as np

from hss.transforms import Resample


def test_resample():
    f = Resample(num=100)
    x = np.array([1, 2, 3, 5])
    y = f(x)

    assert isinstance(y, numpy.ndarray)
    assert y.shape == (100,)

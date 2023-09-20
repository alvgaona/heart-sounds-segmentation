import os

import scipy
from torchvision import transforms

from hss.datasets.heart_sounds import DavidSpringerHSS
from hss.transforms import Resample, FSST

ROOT = os.path.dirname(os.path.dirname(__file__))


def test_in_memory_dataset():
    dataset = DavidSpringerHSS(os.path.join(ROOT, "resources/data"), download=True, in_memory=True)

    assert len(dataset.data) == 792


def test_in_memory_fsst():
    transform = transforms.Compose(
        (
            Resample(35500),
            FSST(
                1000,
                window=scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=True),
                truncate_freq=(25, 200),
                stack=True,
            ),
        )
    )
    dataset = DavidSpringerHSS(os.path.join(ROOT, "resources/data"), download=True, in_memory=True, transform=transform)

    assert len(dataset.data) == 792


import pytest
import scipy
from torchvision import transforms

from hss.datasets.heart_sounds import DavidSpringerHSS
from hss.transforms import FSST, Resample


@pytest.fixture
def fs() -> int:
    return 1000


@pytest.fixture
def dataset_path() -> str:
    return "resources/data"


@pytest.fixture
def transform(fs: int) -> transforms.Compose:
    return transforms.Compose(
        (
            Resample(35500),
            FSST(
                fs=fs,
                window=scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=True),
                truncate_freq=(25, 200),
                stack=True,
            ),
        )
    )


def test_in_memory_dataset(dataset_path: str) -> None:
    dataset = DavidSpringerHSS(dataset_path, download=True, in_memory=True)
    assert len(dataset.data) == 792


def test_in_memory_fsst(dataset_path: str, transform: transforms.Compose) -> None:
    dataset = DavidSpringerHSS(dataset_path, download=True, in_memory=True, transform=transform)
    assert len(dataset.data) == 792


def test_in_memory_framed_fsst(dataset_path: str, transform: transforms.Compose) -> None:
    dataset = DavidSpringerHSS(dataset_path, download=True, in_memory=True, framing=True, transform=transform)
    assert len(dataset.data) == 792 * 33


@pytest.mark.parametrize(
    "in_memory,framing,expected_length",
    [
        (True, False, 792),
        (True, True, 792 * 33),
        (False, False, 792),
        (False, True, 792 * 33),
    ],
)
def test_dataset_configurations(
    dataset_path: str, transform: transforms.Compose, in_memory: bool, framing: bool, expected_length: int
) -> None:
    dataset = DavidSpringerHSS(dataset_path, download=True, in_memory=in_memory, framing=framing, transform=transform)
    assert len(dataset.data) == expected_length

import pytest
import scipy
from torchvision import transforms

from hss.datasets.heart_sounds import DavidSpringerHSS
from hss.transforms import FSST


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
            FSST(
                fs=fs,
                window=scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=False),
            ),
        )
    )


@pytest.mark.parametrize(
    "in_memory,framing,expected_length",
    [
        (True, False, 5),
        (True, True, 5 * 33),
        (False, False, 0),
        (False, True, 0),
    ],
)
def test_dataset_state(
    dataset_path: str, transform: transforms.Compose, in_memory: bool, framing: bool, expected_length: int
) -> None:
    dataset = DavidSpringerHSS(
        dataset_path,
        download=True,
        in_memory=in_memory,
        framing=framing,
        count=5,
        transform=transform,
    )
    assert len(dataset.data) == expected_length


def test_springer_dataset_framing(dataset_path: str, transform: transforms.Compose) -> None:
    dataset = DavidSpringerHSS(
        dataset_path,
        download=True,
        in_memory=True,
        framing=True,
        count=5,
        transform=transform,
        verbose=False,
    )

    for x, y in dataset:
        assert x.shape == (2000, 65)
        assert y.shape == (2000,)


def test_springer_dataset(dataset_path: str, transform: transforms.Compose) -> None:
    dataset = DavidSpringerHSS(
        dataset_path,
        download=True,
        in_memory=True,
        transform=transform,
        verbose=False,
    )

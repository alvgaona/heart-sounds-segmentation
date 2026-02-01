from pathlib import Path

import pytest
import scipy
from torchvision import transforms

from hss.datasets import CirCorDataset, DavidSpringerHSS
from hss.transforms import FSST

DATASET_PATH = Path("resources/data")


@pytest.fixture
def fs() -> int:
    return 1000


@pytest.fixture
def dataset_path() -> str:
    return str(DATASET_PATH)


@pytest.fixture
def transform(fs: int) -> transforms.Compose:
    return transforms.Compose(
        (
            FSST(
                fs=fs,
                window=scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=False),
                truncate_freq=(25, 200),
                stack=True,
            ),
        )
    )


springer_available = pytest.mark.skipif(
    not (DATASET_PATH / "springer").exists(),
    reason="Springer dataset not downloaded",
)
circor_available = pytest.mark.skipif(
    not (DATASET_PATH / "circor").exists(),
    reason="CirCor dataset not downloaded",
)


@springer_available
@pytest.mark.parametrize(
    "in_memory,framing,expected_length",
    [
        (True, False, 5),
        (True, True, None),  # Frame count depends on recording lengths
        (False, False, 0),
        (False, True, 0),
    ],
)
def test_dataset_state(
    dataset_path: str, transform: transforms.Compose, in_memory: bool, framing: bool, expected_length: int | None
) -> None:
    dataset = DavidSpringerHSS(
        dataset_path,
        download=False,
        in_memory=in_memory,
        framing=framing,
        count=5,
        transform=transform,
    )
    if expected_length is None:
        assert len(dataset.data) > 0
    else:
        assert len(dataset.data) == expected_length


@springer_available
def test_springer_dataset_framing(dataset_path: str, transform: transforms.Compose) -> None:
    dataset = DavidSpringerHSS(
        dataset_path,
        download=False,
        in_memory=True,
        framing=True,
        count=5,
        transform=transform,
        verbose=False,
    )

    for x, y in dataset:
        assert x.shape == (2000, 44)
        assert y.shape == (2000,)


@circor_available
def test_circor_dataset_loads(dataset_path: str) -> None:
    dataset = CirCorDataset(
        dataset_path,
        download=False,
        in_memory=True,
        count=5,
        verbose=False,
    )

    assert len(dataset) == 5

    for x, y in dataset:
        assert x.ndim == 1
        assert y.ndim == 1
        assert x.shape[0] == y.shape[0]
        assert y.min() >= -1 and y.max() <= 4  # -1=unannotated, 1=S1, 2=Systole, 3=S2, 4=Diastole


@circor_available
def test_circor_dataset_framing(dataset_path: str, transform: transforms.Compose) -> None:
    dataset = CirCorDataset(
        dataset_path,
        download=False,
        in_memory=True,
        framing=True,
        count=5,
        transform=transform,
        verbose=False,
    )

    assert len(dataset) > 0

    for x, y in dataset:
        assert x.shape == (2000, 44)
        assert y.shape == (2000,)

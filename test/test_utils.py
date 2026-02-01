import torch

from hss.utils.preprocess import frame_signal


def test_frame_1d_signal():
    n = torch.linspace(1, 35000, steps=35000)
    signal = torch.sin(n).t()
    labels = torch.randn((35000,))
    frames, labels = frame_signal(signal, labels, 1000, 2000)

    # floor((35000 - 2000) / 1000) = 33
    assert len(frames) == 33
    assert len(labels) == 33
    for f, l in zip(frames, labels, strict=False):
        assert f.shape == torch.Size([2000, 1])
        assert l.shape == torch.Size([2000, 1])


def test_frame_n_dimensional_signal():
    signal = torch.randn((35000, 2))  # (time, features)
    labels = torch.randn((35000, 1))

    frames, labels = frame_signal(signal, labels, 1000, 2000)

    # floor((35000 - 2000) / 1000) = 33
    assert len(frames) == 33
    assert len(labels) == 33
    for f, l in zip(frames, labels, strict=False):
        assert f.shape == torch.Size([2000, 2])
        assert l.shape == torch.Size([2000, 1])

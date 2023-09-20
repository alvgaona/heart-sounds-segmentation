import torch

from hss.utils.preprocess import frame_signal


def test_frame_1d_signal():
    n = torch.linspace(1, 35000, steps=35000)
    signal = torch.sin(n).t()
    labels = torch.randn((35000,))
    frames, labels = frame_signal(signal, labels, 1000, 2000)

    assert len(frames) == 32
    assert len(labels) == 32
    for f, l in zip(frames, labels):
        assert f.shape == torch.Size([2000, 1])
        assert l.shape == torch.Size([2000, 1])


def test_frame_n_dimensional_signal():
    signal = torch.randn((2, 35000))
    labels = torch.randn((35000, 1))

    frames, labels = frame_signal(signal, labels, 1000, 2000)

    assert len(frames) == 32
    assert len(labels) == 32
    for f, l in zip(frames, labels):
        assert f.shape == torch.Size([2000, 2])
        assert l.shape == torch.Size([2000, 2])

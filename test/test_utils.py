import torch

from hss.utils.preprocess import frame_signal


def test_frame_signals():
    n = torch.linspace(1, 35000, steps=35000)
    signal = torch.sin(n).reshape(-1, 1)
    labels = torch.randn((35000, 1))
    frames, labels = frame_signal(signal, labels, 1000, 2000)

    assert len(frames) == 32
    assert len(labels) == 32
    for f, l in zip(frames, labels):
        assert f.shape == torch.Size([2000, 1])
        assert l.shape == torch.Size([2000, 1])

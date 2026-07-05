"""Smoke tests for the xLSTM emitter + CRF segmenter (Experiment A wiring)."""

import torch

from hss.model.segmenter_crf import HeartSoundSegmenterCRF
from hss.model.segmenter_xlstm import HeartSoundSegmenterXLSTMCRF


INPUT_SIZE = 44  # FSST feature dimension


def _model(**kw):
    return HeartSoundSegmenterXLSTMCRF(input_size=INPUT_SIZE, hidden_size=48, num_heads=4, **kw)


def test_forward_emissions_shape():
    model = _model()
    x = torch.randn(3, 60, INPUT_SIZE)
    emissions = model(x)
    assert emissions.shape == (3, 60, 4)
    assert torch.isfinite(emissions).all()


def test_loss_is_scalar_and_backprops():
    model = _model()
    x = torch.randn(2, 50, INPUT_SIZE)
    tags = torch.randint(0, 4, (2, 50))
    loss = model.loss(x, tags)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_decode_and_valid_shapes_and_ranges():
    model = _model()
    x = torch.randn(2, 40, INPUT_SIZE)
    decoded = model.decode(x)
    assert torch.as_tensor(decoded).shape == (2, 40)

    valid = model.decode_valid(x)
    assert valid.shape == (2, 40)
    assert int(valid.min()) >= 0 and int(valid.max()) <= 3  # labels 0-3


def test_no_batch_size_tied_buffers():
    """Unlike the BiLSTM model, the xLSTM segmenter must accept any batch size (no h0/c0)."""
    model = _model(batch_size=2)
    for batch in (1, 5, 8):
        out = model(torch.randn(batch, 30, INPUT_SIZE))
        assert out.shape == (batch, 30, 4)


def test_param_count_reference_vs_bilstm(capsys):
    """Report params for the iso-parameter comparison (BiLSTM-240 vs xLSTM-240)."""

    def n(m):
        return sum(p.numel() for p in m.parameters())

    bilstm = HeartSoundSegmenterCRF(input_size=INPUT_SIZE, hidden_size=240)
    xlstm = HeartSoundSegmenterXLSTMCRF(input_size=INPUT_SIZE, hidden_size=240, num_heads=4)
    with capsys.disabled():
        print(f"\n  params  BiLSTM-240={n(bilstm):,}  xLSTM-240={n(xlstm):,}")
    assert n(xlstm) > 0

"""Self-consistency + interface tests for the vendored mLSTM core (Tier-1 validation).

The key correctness check is that the parallel form and the recurrent step form compute the same
thing — that both validates the math (independent of the official package) and confirms the
streaming path used by Experiment B. Numerical parity against the official ``xlstm`` backend is a
separate GPU test built from ``scripts/xlstm/probe_official_xlstm.py``.
"""

import torch

from hss.model.xlstm import (
    BidirectionalXLSTMEncoder,
    CausalXLSTMEncoder,
    PhaseBidirectionalXLSTMEncoder,
    PhaseClockMLSTMLayer,
    mLSTMLayer,
    mlstm_init_state,
    mlstm_parallel,
    mlstm_step,
)


def _random_qkvif(b, nh, t, dh, *, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, nh, t, dh, generator=g, dtype=dtype)
    k = torch.randn(b, nh, t, dh, generator=g, dtype=dtype)
    v = torch.randn(b, nh, t, dh, generator=g, dtype=dtype)
    igate = torch.randn(b, nh, t, 1, generator=g, dtype=dtype)
    fgate = torch.randn(b, nh, t, 1, generator=g, dtype=dtype) + 2.0  # bias toward remembering
    return q, k, v, igate, fgate


def test_parallel_equals_recurrent():
    """The O(T^2) parallel form must equal a sequential scan of the O(1) step form.

    Checked with ``eps=0`` so this is a true mathematical-equivalence assertion (machine epsilon).
    The default ``eps`` is a float32 safety floor and perturbs the two forms identically to ~eps
    (verified: the residual scales linearly with eps and vanishes at eps=0, T-independent).
    """
    b, nh, t, dh = 2, 3, 64, 8
    q, k, v, igate, fgate = _random_qkvif(b, nh, t, dh)

    h_par = mlstm_parallel(q, k, v, igate, fgate, eps=0.0)  # (B, NH, T, DH)

    state = mlstm_init_state(b, nh, dh, dtype=torch.float64)
    h_steps = []
    for i in range(t):
        h_t, state = mlstm_step(
            state,
            q[:, :, i : i + 1].transpose(-2, -1),  # (B, NH, DH, 1)
            k[:, :, i : i + 1].transpose(-2, -1),
            v[:, :, i : i + 1].transpose(-2, -1),
            igate[:, :, i : i + 1],  # (B, NH, 1, 1)
            fgate[:, :, i : i + 1],
            eps=0.0,
        )
        h_steps.append(h_t.transpose(-2, -1))  # (B, NH, 1, DH)
    h_rec = torch.cat(h_steps, dim=-2)  # (B, NH, T, DH)

    max_diff = (h_par - h_rec).abs().max().item()
    assert max_diff < 1e-11, f"parallel vs recurrent mismatch: {max_diff}"


def test_parallel_is_causal():
    """Output at step t must not depend on inputs after t (lower-triangular decay mask)."""
    b, nh, t, dh = 1, 2, 12, 8
    q, k, v, igate, fgate = _random_qkvif(b, nh, t, dh, seed=1)
    h1 = mlstm_parallel(q, k, v, igate, fgate)

    # perturb only the last time step
    v2 = v.clone()
    v2[:, :, -1] += 5.0
    h2 = mlstm_parallel(q, k, v2, igate, fgate)

    # everything before the last step is unchanged
    assert torch.allclose(h1[:, :, :-1], h2[:, :, :-1], atol=1e-10)


def test_gradcheck_parallel():
    """Analytic gradients of the parallel operator are correct (double precision)."""
    b, nh, t, dh = 1, 2, 6, 4
    q, k, v, igate, fgate = _random_qkvif(b, nh, t, dh, seed=2)
    for tensor in (q, k, v, igate, fgate):
        tensor.requires_grad_(True)
    assert torch.autograd.gradcheck(mlstm_parallel, (q, k, v, igate, fgate), atol=1e-5, rtol=1e-3)


def test_layer_shape_and_finite():
    layer = mLSTMLayer(input_size=44, hidden_size=48, num_heads=4)
    x = torch.randn(2, 20, 44)
    y = layer(x)
    assert y.shape == (2, 20, 48)
    assert torch.isfinite(y).all()


def test_bidirectional_encoder_dropin_shape():
    """Encoder must emit (B, T, 2*hidden) so the existing Linear(2H->4) head is unchanged."""
    enc = BidirectionalXLSTMEncoder(input_size=44, hidden_size=240, num_heads=4, num_layers=2)
    x = torch.randn(2, 100, 44)
    y = enc(x)
    assert y.shape == (2, 100, 480)
    assert torch.isfinite(y).all()


def test_causal_encoder_shape():
    """Causal encoder is half the width (no backward pass): (B, T, hidden)."""
    enc = CausalXLSTMEncoder(input_size=44, hidden_size=240, num_heads=4, num_layers=2)
    y = enc(torch.randn(2, 100, 44))
    assert y.shape == (2, 100, 240)
    assert torch.isfinite(y).all()


def test_causal_forward_equals_streaming_steps():
    """Parallel causal forward must equal the O(1)/frame streaming step chain (the deployability path)."""
    enc = CausalXLSTMEncoder(input_size=8, hidden_size=16, num_heads=4, num_layers=2).double().eval()
    x = torch.randn(2, 32, 8, dtype=torch.float64)

    y_par = enc(x)  # (B, T, hidden)

    states = enc.init_state(2, dtype=torch.float64)
    outs = []
    for t in range(x.shape[1]):
        h, states = enc.step(x[:, t], states)
        outs.append(h)
    y_stream = torch.stack(outs, dim=1)

    # Agreement is at the eps-floor level (~1e-6): the layer uses the default eps=1e-6, and parallel
    # vs recurrent differ by ~eps (see test_parallel_equals_recurrent, exact at eps=0). A projection
    # bug would show an O(1) mismatch, not O(eps) — so this confirms the streaming wiring is correct.
    max_diff = (y_par - y_stream).abs().max().item()
    assert max_diff < 1e-5, f"causal forward vs streaming mismatch: {max_diff}"


def test_phase_clock_forward_equals_streaming():
    """Phase-clock layer: parallel forward (cumsum phase) == streaming step (running-sum phase)."""
    layer = PhaseClockMLSTMLayer(input_size=8, hidden_size=16, num_heads=4, n_harmonics=2).double().eval()
    x = torch.randn(2, 24, 8, dtype=torch.float64)

    y_par = layer(x)

    state = layer.init_state(2, dtype=torch.float64)
    outs = []
    for t in range(x.shape[1]):
        h, state = layer.step(x[:, t], state)
        outs.append(h)
    y_stream = torch.stack(outs, dim=1)

    assert (y_par - y_stream).abs().max().item() < 1e-5


def test_phase_encoder_shape_and_finite():
    enc = PhaseBidirectionalXLSTMEncoder(input_size=44, hidden_size=48, num_heads=4, num_layers=2)
    y = enc(torch.randn(2, 60, 44))
    assert y.shape == (2, 60, 96)
    assert torch.isfinite(y).all()

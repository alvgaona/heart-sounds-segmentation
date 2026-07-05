"""Self-consistency + interface tests for the vendored mLSTM core (Tier-1 validation).

The key correctness check is that the parallel form and the recurrent step form compute the same
thing — that both validates the math (independent of the official package) and confirms the
streaming path used by Experiment B. Numerical parity against the official ``xlstm`` backend is a
separate GPU test built from ``scripts/xlstm/probe_official_xlstm.py``.
"""

import torch

from hss.model.xlstm import (
    BidirectionalXLSTMEncoder,
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

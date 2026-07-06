"""Vendored minimal mLSTM (matrix-LSTM) core for the xLSTM emitter experiment.

Implements the stabilized mLSTM operation from Beck et al. 2024 ("xLSTM: Extended Long
Short-Term Memory", NeurIPS 2024) in pure PyTorch, in two numerically-equivalent forms:

- ``mlstm_parallel``  — the O(T^2) parallel form used for training.
- ``mlstm_step``      — the O(1)/step recurrent form used for streaming (Experiment B).

Their equivalence (see ``tests/test_xlstm.py``) both validates the implementation and gives us
the constant-memory streaming path. Numerical parity against the official ``xlstm`` package's
``parallel_stabilized_simple`` backend is checked separately on GPU (see
``scripts/xlstm/probe_official_xlstm.py`` + the parity test) — this core targets that *operator*
(q, k, v, input-gate, forget-gate -> h), not the full LM block.

Why vendored rather than the official package: this runs identically on CPU/MPS/CUDA (the repo
defaults to CPU; the official sLSTM fast path is CUDA-only), quantizes/exports cleanly for the C8
deployability story, and exposes the explicit per-step recurrence Experiment B needs.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


MLSTMState = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _bias_linspace_init_(bias: torch.Tensor, start: float, end: float) -> None:
    """Fill a 1-D bias with evenly spaced values in ``[start, end]`` (official xLSTM init)."""
    with torch.no_grad():
        bias.copy_(torch.linspace(start, end, bias.numel()))


def mlstm_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Stabilized parallel mLSTM operator.

    Args:
        q, k, v: ``(B, NH, T, DH)`` per-head query/key/value.
        igate_preact, fgate_preact: ``(B, NH, T, 1)`` input/forget gate pre-activations.
        eps: numerical floor for the normalizer.

    Returns:
        ``(B, NH, T, DH)`` hidden states, before any output gate.
    """
    _, _, T, DH = q.shape

    log_f = F.logsigmoid(fgate_preact)  # (B, NH, T, 1)
    # cumulative forget mass with a leading zero so differences give inclusive sums
    log_f_cumsum = torch.cat([torch.zeros_like(log_f[:, :, :1, :]), log_f.cumsum(dim=-2)], dim=-2)  # (B,NH,T+1,1)

    # log_fg[i, j] = sum_{s=j+1..i} log_f_s  (drop the padded row/col), masked to i >= j
    log_fg = (log_f_cumsum - log_f_cumsum.transpose(-2, -1))[:, :, 1:, 1:]  # (B, NH, T, T)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
    log_fg = log_fg.masked_fill(~causal, float("-inf"))

    # add the input gate at the source position j (broadcast over rows i)
    log_d = log_fg + igate_preact.transpose(-2, -1)  # (B, NH, T, T)
    m, _ = log_d.max(dim=-1, keepdim=True)  # (B, NH, T, 1) row-wise stabilizer
    d = torch.exp(log_d - m)  # (B, NH, T, T)

    qk = (q @ k.transpose(-2, -1)) / math.sqrt(DH)  # (B, NH, T, T)
    c = qk * d
    denom = torch.maximum(c.sum(dim=-1, keepdim=True).abs(), torch.exp(-m))  # (B, NH, T, 1)
    c = c / (denom + eps)
    return c @ v  # (B, NH, T, DH)


def mlstm_init_state(batch: int, num_heads: int, head_dim: int, *, device=None, dtype=torch.float32) -> MLSTMState:
    """Zero cell (C), normalizer (n) and stabilizer (m) states for the recurrent form."""
    c = torch.zeros(batch, num_heads, head_dim, head_dim, device=device, dtype=dtype)
    n = torch.zeros(batch, num_heads, head_dim, 1, device=device, dtype=dtype)
    m = torch.zeros(batch, num_heads, 1, 1, device=device, dtype=dtype)
    return c, n, m


def mlstm_step(
    state: MLSTMState,
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    igate_t: torch.Tensor,
    fgate_t: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, MLSTMState]:
    """One stabilized recurrent mLSTM step (O(1) memory).

    Args:
        state: ``(C, n, m)`` from ``mlstm_init_state`` or a previous step.
        q_t, k_t, v_t: ``(B, NH, DH, 1)`` per-head vectors at this step.
        igate_t, fgate_t: ``(B, NH, 1, 1)`` gate pre-activations at this step.

    Returns:
        ``(h_t, new_state)`` with ``h_t`` of shape ``(B, NH, DH, 1)`` (before output gate).
    """
    c_prev, n_prev, m_prev = state
    dh = q_t.shape[-2]

    log_f = F.logsigmoid(fgate_t)  # (B, NH, 1, 1)
    m_new = torch.maximum(log_f + m_prev, igate_t)  # (B, NH, 1, 1)
    i_stab = torch.exp(igate_t - m_new)
    f_stab = torch.exp(log_f + m_prev - m_new)

    c = f_stab * c_prev + i_stab * (v_t @ k_t.transpose(-2, -1))  # (B, NH, DH, DH)
    n = f_stab * n_prev + i_stab * k_t  # (B, NH, DH, 1)

    q_scaled = q_t / math.sqrt(dh)
    num = c @ q_scaled  # (B, NH, DH, 1)
    denom = torch.maximum((n.transpose(-2, -1) @ q_scaled).abs(), torch.exp(-m_new))  # (B, NH, 1, 1)
    h = num / (denom + eps)
    return h, (c, n, m_new)


class mLSTMLayer(nn.Module):
    """A single multi-head mLSTM layer: input projections -> mLSTM operator -> output gate.

    Causal (unidirectional). Wrap two of these (forward + reversed) for a bidirectional encoder.
    """

    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 4) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q = nn.Linear(input_size, hidden_size)
        self.k = nn.Linear(input_size, hidden_size)
        self.v = nn.Linear(input_size, hidden_size)
        self.igate = nn.Linear(input_size, num_heads)
        self.fgate = nn.Linear(input_size, num_heads)
        self.ogate = nn.Linear(input_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.reset_gate_parameters()

    def reset_gate_parameters(self) -> None:
        """Official xLSTM gate init: zero the gate weights (input-independent at init) and bias the
        forget gate toward remembering (linspace 3->6 across heads => retention ~0.95-0.998), so the
        model starts with long memory instead of a ~1-step half-life. See Beck et al. 2024.
        """
        nn.init.zeros_(self.fgate.weight)
        _bias_linspace_init_(self.fgate.bias, 3.0, 6.0)
        nn.init.zeros_(self.igate.weight)
        nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)

    def _project(self, x: torch.Tensor):
        b, t, _ = x.shape
        nh, dh = self.num_heads, self.head_dim

        def heads(proj: torch.Tensor) -> torch.Tensor:
            return proj.view(b, t, nh, dh).transpose(1, 2)  # (B, NH, T, DH)

        q, k, v = heads(self.q(x)), heads(self.k(x)), heads(self.v(x))
        ig = self.igate(x).permute(0, 2, 1).unsqueeze(-1)  # (B, NH, T, 1)
        fg = self.fgate(x).permute(0, 2, 1).unsqueeze(-1)  # (B, NH, T, 1)
        return q, k, v, ig, fg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Parallel forward. ``x``: (B, T, input_size) -> (B, T, hidden_size)."""
        b, t, _ = x.shape
        q, k, v, ig, fg = self._project(x)
        h = mlstm_parallel(q, k, v, ig, fg)  # (B, NH, T, DH)
        h = h.transpose(1, 2).reshape(b, t, self.hidden_size)  # (B, T, H)
        h = self.norm(h) * torch.sigmoid(self.ogate(x))
        return h

    def init_state(self, batch: int, *, device=None, dtype=torch.float32) -> MLSTMState:
        """Zero recurrent state for streaming (Experiment B)."""
        return mlstm_init_state(batch, self.num_heads, self.head_dim, device=device, dtype=dtype)

    def step(self, x_t: torch.Tensor, state: MLSTMState) -> tuple[torch.Tensor, MLSTMState]:
        """One O(1) streaming step. ``x_t``: (B, input_size) -> h_t: (B, hidden_size), new state.

        Numerically equal to the causal ``forward`` at the same position (validated in tests).
        """
        b = x_t.shape[0]
        nh, dh = self.num_heads, self.head_dim
        q_t = self.q(x_t).view(b, nh, dh, 1)  # same head split as _project, one frame
        k_t = self.k(x_t).view(b, nh, dh, 1)
        v_t = self.v(x_t).view(b, nh, dh, 1)
        ig = self.igate(x_t).view(b, nh, 1, 1)
        fg = self.fgate(x_t).view(b, nh, 1, 1)
        h_t, new_state = mlstm_step(state, q_t, k_t, v_t, ig, fg)  # (B, NH, DH, 1)
        h = h_t.reshape(b, self.hidden_size)
        h = self.norm(h) * torch.sigmoid(self.ogate(x_t))
        return h, new_state


class BidirectionalXLSTMEncoder(nn.Module):
    """Drop-in replacement for the 2-layer BiLSTM emitter core.

    Produces ``(B, T, 2 * hidden_size)`` so the existing ``Linear(2H -> 4)`` classification /
    CRF-emission head is unchanged. Each layer runs a forward and a reversed ``mLSTMLayer`` and
    concatenates them (mirroring ``bidirectional=True`` in ``nn.LSTM``).
    """

    def __init__(self, input_size: int, hidden_size: int = 240, num_heads: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        dims = [input_size] + [2 * hidden_size] * (num_layers - 1)
        self.fwd = nn.ModuleList(mLSTMLayer(d, hidden_size, num_heads) for d in dims)
        self.bwd = nn.ModuleList(mLSTMLayer(d, hidden_size, num_heads) for d in dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fwd, bwd in zip(self.fwd, self.bwd, strict=True):
            f = fwd(x)
            b = bwd(x.flip(1)).flip(1)
            x = torch.cat([f, b], dim=-1)
        return x


class CausalXLSTMEncoder(nn.Module):
    """Unidirectional (causal) mLSTM stack for streaming inference (Experiment B).

    Emits ``(B, T, hidden_size)`` — half the width of the bidirectional encoder, since there is no
    backward pass. ``forward`` (parallel, for training) and a chained ``step`` (O(1)/frame, constant
    memory) are numerically identical; the streaming path enables real-time, on-device segmentation.
    """

    def __init__(self, input_size: int, hidden_size: int = 240, num_heads: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        dims = [input_size] + [hidden_size] * (num_layers - 1)
        self.layers = nn.ModuleList(mLSTMLayer(d, hidden_size, num_heads) for d in dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def init_state(self, batch: int, *, device=None, dtype=torch.float32) -> list[MLSTMState]:
        return [layer.init_state(batch, device=device, dtype=dtype) for layer in self.layers]

    def step(self, x_t: torch.Tensor, states: list[MLSTMState]) -> tuple[torch.Tensor, list[MLSTMState]]:
        """One streaming frame: ``x_t`` (B, input_size) through all layers -> (B, hidden_size), new states."""
        new_states: list[MLSTMState] = []
        h = x_t
        for layer, st in zip(self.layers, states, strict=True):
            h, ns = layer.step(h, st)
            new_states.append(ns)
        return h, new_states


PhaseState = tuple[MLSTMState, torch.Tensor]  # (inner mLSTM state, phase clock φ)


class PhaseClockMLSTMLayer(nn.Module):
    """mLSTM layer with a learned monotonic cardiac-phase clock (Experiment C).

    A per-frame nonnegative rate ``softplus(rate(x))`` accumulates into a phase ``φ_t = Σ rate`` (cumsum
    in the parallel form, running sum in the streaming ``step``). The phase harmonics
    ``[sin kφ, cos kφ]`` are concatenated to the input so the mLSTM gates get an explicit periodic /
    duration-aware inductive bias aligned to the heartbeat. Wraps a plain ``mLSTMLayer`` on the
    phase-augmented input, so it inherits the validated parallel==streaming operator.
    """

    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 4, n_harmonics: int = 2) -> None:
        super().__init__()
        self.n_harmonics = n_harmonics
        self.hidden_size = hidden_size
        self.rate = nn.Linear(input_size, 1)
        self.inner = mLSTMLayer(input_size + 2 * n_harmonics, hidden_size, num_heads)

    def _phase_features(self, phi: torch.Tensor) -> torch.Tensor:
        """φ (..., ) -> (..., 2*n_harmonics) = [sin kφ, cos kφ]_{k=1..H}."""
        ks = torch.arange(1, self.n_harmonics + 1, device=phi.device, dtype=phi.dtype)
        ang = phi.unsqueeze(-1) * ks
        return torch.cat([ang.sin(), ang.cos()], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Parallel forward. ``x``: (B, T, input_size) -> (B, T, hidden_size)."""
        rate = F.softplus(self.rate(x)).squeeze(-1)  # (B, T) nonnegative phase increments
        phi = torch.cumsum(rate, dim=1)  # (B, T) accumulated cardiac phase
        pe = self._phase_features(phi)  # (B, T, 2H)
        return self.inner(torch.cat([x, pe], dim=-1))

    def init_state(self, batch: int, *, device=None, dtype=torch.float32) -> PhaseState:
        inner = self.inner.init_state(batch, device=device, dtype=dtype)
        return inner, torch.zeros(batch, device=device, dtype=dtype)

    def step(self, x_t: torch.Tensor, state: PhaseState) -> tuple[torch.Tensor, PhaseState]:
        """One streaming frame: advance the phase clock, then the inner mLSTM step."""
        inner_state, phi = state
        phi = phi + F.softplus(self.rate(x_t)).squeeze(-1)  # (B,)
        pe = self._phase_features(phi)  # (B, 2H)
        h, new_inner = self.inner.step(torch.cat([x_t, pe], dim=-1), inner_state)
        return h, (new_inner, phi)


class PhaseBidirectionalXLSTMEncoder(nn.Module):
    """Bidirectional stack of phase-clock mLSTM layers (Experiment C). Emits ``(B, T, 2*hidden_size)``."""

    def __init__(
        self, input_size: int, hidden_size: int = 240, num_heads: int = 4, num_layers: int = 2, n_harmonics: int = 2
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        dims = [input_size] + [2 * hidden_size] * (num_layers - 1)
        self.fwd = nn.ModuleList(PhaseClockMLSTMLayer(d, hidden_size, num_heads, n_harmonics) for d in dims)
        self.bwd = nn.ModuleList(PhaseClockMLSTMLayer(d, hidden_size, num_heads, n_harmonics) for d in dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fwd, bwd in zip(self.fwd, self.bwd, strict=True):
            f = fwd(x)
            b = bwd(x.flip(1)).flip(1)
            x = torch.cat([f, b], dim=-1)
        return x

"""Streaming (fixed-lag) constrained Viterbi for real-time valid-cycle segmentation (Experiment B).

The offline ``decode_valid`` needs the whole recording (forward-backward posterior). For streaming we
run a causal, fixed-lag Viterbi over per-frame emission scores with a hard cardiac-cycle transition
mask — only self-loops and S1->Systole->S2->Diastole->S1 are allowed — so any single decoded trellis
path is a valid cycle. Frame ``p`` is committed at time ``p + lag`` by backtracking from the best state
then; as ``lag -> T`` the output converges exactly to the offline constrained Viterbi. Trellis update is
O(states^2) per frame and needs only the last ``lag`` backpointer columns, so memory is constant in T.
"""

from __future__ import annotations

import torch


NEG_INF = -1e9
# i -> j allowed transitions, 0-indexed (S1=0, Systole=1, S2=2, Diastole=3): self-loops + forward cycle.
VALID_TRANSITIONS = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 0)]


def transition_logmask(num_states: int = 4, device: torch.device | None = None) -> torch.Tensor:
    """(S, S) log-mask: 0 for an allowed i->j transition, NEG_INF otherwise."""
    m = torch.full((num_states, num_states), NEG_INF, device=device)
    for i, j in VALID_TRANSITIONS:
        m[i, j] = 0.0
    return m


def _forward_trellis(emissions: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
    """Run the constrained Viterbi forward pass; return backpointers (T, S) and best state per frame."""
    t_len, n = emissions.shape
    trans = transition_logmask(n, emissions.device)
    score = emissions[0].clone()
    back = torch.zeros(t_len, n, dtype=torch.long, device=emissions.device)
    best_at = [int(score.argmax())]
    for t in range(1, t_len):
        cand = score.unsqueeze(1) + trans  # (S_prev, S_next)
        bp = cand.argmax(0)
        score = cand.gather(0, bp.unsqueeze(0)).squeeze(0) + emissions[t]
        back[t] = bp
        best_at.append(int(score.argmax()))
    return back, best_at


def constrained_viterbi(emissions: torch.Tensor) -> torch.Tensor:
    """Offline constrained Viterbi over (T, S) emission log-scores -> (T,) guaranteed-valid state path."""
    back, best_at = _forward_trellis(emissions)
    t_len = emissions.shape[0]
    path = torch.zeros(t_len, dtype=torch.long, device=emissions.device)
    path[-1] = best_at[-1]
    for t in range(t_len - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def stream_decode(emissions: torch.Tensor, lag: int) -> torch.Tensor:
    """Fixed-lag constrained Viterbi simulating streaming. (T, S) -> (T,) states.

    Frame ``p`` is committed at time ``min(p + lag, T-1)`` by backtracking to it from the best state
    then. ``lag >= T-1`` reproduces the offline Viterbi exactly.
    """
    back, best_at = _forward_trellis(emissions)
    t_len = emissions.shape[0]
    out = torch.zeros(t_len, dtype=torch.long, device=emissions.device)
    for p in range(t_len):
        d = min(p + lag, t_len - 1)
        s = best_at[d]
        for u in range(d, p, -1):
            s = int(back[u, s])
        out[p] = s
    return out


def valid_transition_fraction(path: torch.Tensor) -> float:
    """Fraction of consecutive (t, t+1) transitions in ``path`` that are cardiac-valid (1.0 = all valid)."""
    allowed = set(VALID_TRANSITIONS)
    seq = path.tolist()
    if len(seq) < 2:
        return 1.0
    ok = sum((a, b) in allowed for a, b in zip(seq[:-1], seq[1:], strict=True))
    return ok / (len(seq) - 1)

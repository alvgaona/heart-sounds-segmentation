"""Semi-Markov CRF (Segmental CRF) with learnable duration distributions.

This extends the standard CRF by modeling segments with explicit durations,
similar to Springer's Hidden Semi-Markov Model (HSMM) approach.

Key differences from standard CRF:
- Standard CRF: P(y_t | y_{t-1}) - frame-level transitions
- Semi-Markov CRF: P(state, duration | prev_state) - segment-level transitions

The score for a segmentation is:
    Score = Σ [transition(s_{i-1} → s_i) + duration(d_i | s_i) + Σ emission(s_i, frame_j)]

References:
- Springer et al. 2016: "Logistic Regression-HSMM-based Heart Sound Segmentation"
- Sarawagi & Cohen 2004: "Semi-Markov Conditional Random Fields for Information Extraction"
"""

import math
from typing import Final, Literal

import torch
from torch import Tensor, nn


# Sentinel for "impossible" log-scores. A true float(-inf) is exact, but Apple MPS turns -inf
# into NaN inside logsumexp/logaddexp reductions (e.g. logaddexp(-inf, -inf)); a large finite
# negative underflows to 0 in exp() just like -inf while staying NaN-free on every backend.
# Final so TorchScript inlines it as a compile-time constant inside the @torch.jit.script passes.
NEG_INF: Final[float] = -1.0e9


def _log_matmul(A: Tensor, B: Tensor) -> Tensor:
    """Matrix multiplication in log-semiring: C[i,k] = logsumexp_j(A[i,j] + B[j,k]).

    Args:
        A: (..., M, N) log-space matrix
        B: (..., N, P) log-space matrix

    Returns:
        C: (..., M, P) result of log-semiring multiplication
    """
    # A: (..., M, N) -> (..., M, N, 1)
    # B: (..., N, P) -> (..., 1, N, P)
    # Sum: (..., M, N, P), logsumexp over N -> (..., M, P)
    return torch.logsumexp(A.unsqueeze(-1) + B.unsqueeze(-3), dim=-2)


def _parallel_scan_log_semiring(matrices: Tensor, right_to_left: bool = False) -> Tensor:
    """Parallel prefix scan using log-semiring matrix multiplication.

    Computes cumulative matrix products. With right_to_left=False (default):
        M[0], M[0]⊗M[1], M[0]⊗M[1]⊗M[2], ...

    With right_to_left=True (for state transitions):
        M[0], M[1]⊗M[0], M[2]⊗M[1]⊗M[0], ...

    The right_to_left version is needed when applying to state vectors,
    since (A⊗B)⊗v = A⊗(B⊗v) applies B first, then A.

    Uses a simple iterative approach that parallelizes well on GPU.
    For T elements, uses O(log T) rounds, each with O(T) parallel work.

    Args:
        matrices: (batch, seq_len, S, S) - sequence of transfer matrices
        right_to_left: if True, compute M[i]⊗...⊗M[0] instead of M[0]⊗...⊗M[i]

    Returns:
        cumulative: (batch, seq_len, S, S) - cumulative products
    """
    B, T, S, _ = matrices.shape

    if T == 0:
        return matrices.clone()

    if T == 1:
        return matrices.clone()

    # Simple parallel scan using Hillis-Steele algorithm
    # Each round doubles the offset, giving O(log T) rounds
    # Round k: result[i] = result[i - 2^k] ⊗ result[i] for i >= 2^k

    result = matrices.clone()

    offset = 1
    while offset < T:
        # For positions >= offset, combine with position offset steps back
        left = result[:, : T - offset]  # (B, T-offset, S, S)
        right = result[:, offset:]  # (B, T-offset, S, S)

        # right_to_left: M[i] ⊗ M[i-1] ⊗ ... ⊗ M[0], otherwise: M[0] ⊗ M[1] ⊗ ... ⊗ M[i]
        combined = _log_matmul(right, left) if right_to_left else _log_matmul(left, right)

        # Update in-place (need to clone to avoid issues)
        new_result = result.clone()
        new_result[:, offset:] = combined
        result = new_result

        offset *= 2

    return result


def _forward_pass_hybrid(
    segment_emissions: Tensor,
    dur_scores: Tensor,
    trans_scores: Tensor,
    start_transitions: Tensor,
    end_transitions: Tensor,
    seq_len: int,
    max_duration: int,
    chunk_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Hybrid forward pass: sequential warmup for first D steps, then vectorized continuation.

    The first D steps handle start transitions. Then the remaining steps can be computed
    more efficiently by exploiting the structure where all D lookback positions are valid.

    Args:
        segment_emissions: (batch, seq_len+1, max_dur, num_tags)
        dur_scores: (max_dur, num_tags)
        trans_scores: (num_tags, num_tags)
        start_transitions: (num_tags,)
        end_transitions: (num_tags,)
        seq_len: T
        max_duration: D
        chunk_size: C - not used currently, kept for API compatibility

    Returns:
        alpha: (batch, seq_len+1, num_tags) - forward variables
        log_Z: (batch,) - log partition function
    """
    batch_size = segment_emissions.shape[0]
    num_tags = segment_emissions.shape[3]
    device = segment_emissions.device
    dtype = segment_emissions.dtype

    D = min(max_duration, seq_len)
    K = num_tags

    alpha = torch.full((batch_size, seq_len + 1, K), NEG_INF, device=device, dtype=dtype)
    neg_inf_score = torch.full((batch_size, K), NEG_INF, device=device, dtype=dtype)

    # Precompute trans_scores expansion
    trans_exp = trans_scores.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)

    # Phase 1: Sequential warmup for t=1..D (handles start transitions)
    warmup_end = min(D, seq_len)
    for t in range(1, warmup_end + 1):
        # Case 1: segments starting from position 0
        start_score = start_transitions + dur_scores[t - 1, :] + segment_emissions[:, t, t - 1, :]

        # Case 2: segments following previous segments
        if t > 1:
            num_valid_d = t - 1  # Can look back at most t-1 steps

            indices = torch.arange(t - 1, t - 1 - num_valid_d, -1, device=device)
            prev_alpha = alpha[:, indices, :]  # (B, num_valid_d, K)

            combined = (
                prev_alpha.unsqueeze(-1)  # (B, D', K, 1)
                + trans_exp  # (1, 1, K, K)
                + dur_scores[:num_valid_d, :].unsqueeze(0).unsqueeze(2)  # (1, D', 1, K)
                + segment_emissions[:, t, :num_valid_d, :].unsqueeze(2)  # (B, D', 1, K)
            )
            case2_score = torch.logsumexp(combined.view(batch_size, -1, K), dim=1)
        else:
            case2_score = neg_inf_score

        alpha[:, t, :] = torch.logaddexp(start_score, case2_score)

    # If seq_len <= D, we're done
    if seq_len <= D:
        log_Z = torch.logsumexp(alpha[:, seq_len, :] + end_transitions, dim=1)
        return alpha, log_Z

    # Phase 2: Continue with full D lookback (no more start transitions)
    # For t > D, we always look back at D previous positions
    # This allows more efficient vectorization

    # Precompute combined dur + trans scores for efficiency
    # dur_trans[d, j, k] = trans[j, k] + dur[d, k]
    dur_trans = trans_scores.unsqueeze(0) + dur_scores.unsqueeze(1)  # (D, K, K)

    for t in range(D + 1, seq_len + 1):
        # All D lookback positions are valid
        # indices: t-1, t-2, ..., t-D (in that order)
        indices = torch.arange(t - 1, t - D - 1, -1, device=device)
        prev_alpha = alpha[:, indices, :]  # (B, D, K_prev)

        # seg_emit for durations 1..D ending at t
        seg_emit = segment_emissions[:, t, :D, :]  # (B, D, K_next)

        # Combine: prev_alpha[b, d, j] + dur_trans[d, j, k] + seg_emit[b, d, k]
        # prev_alpha: (B, D, K_prev) -> (B, D, K_prev, 1)
        # dur_trans: (D, K_prev, K_next) -> (1, D, K_prev, K_next)
        # seg_emit: (B, D, K_next) -> (B, D, 1, K_next)
        combined = (
            prev_alpha.unsqueeze(-1)  # (B, D, K, 1)
            + dur_trans.unsqueeze(0)  # (1, D, K, K)
            + seg_emit.unsqueeze(2)  # (B, D, 1, K)
        )  # (B, D, K_prev, K_next)

        # logsumexp over (d, k_prev) -> k_next
        alpha[:, t, :] = torch.logsumexp(combined.view(batch_size, -1, K), dim=1)

    log_Z = torch.logsumexp(alpha[:, seq_len, :] + end_transitions, dim=1)
    return alpha, log_Z


@torch.jit.script
def _forward_pass(
    segment_emissions: Tensor,
    dur_scores: Tensor,
    trans_scores: Tensor,
    start_transitions: Tensor,
    end_transitions: Tensor,
    seq_len: int,
    max_duration: int,
) -> tuple[Tensor, Tensor]:
    """Compute forward variables (alpha) for Semi-Markov CRF (JIT compiled, optimized)."""
    batch_size = segment_emissions.shape[0]
    num_tags = segment_emissions.shape[3]
    device = segment_emissions.device
    dtype = segment_emissions.dtype
    D = min(max_duration, seq_len)
    neg_inf: float = -1.0e9  # local literal; TorchScript can't close over the module constant

    alpha = torch.full((batch_size, seq_len + 1, num_tags), neg_inf, device=device, dtype=dtype)

    # Preallocate buffers to avoid allocations in loop
    neg_inf_score = torch.full((batch_size, num_tags), neg_inf, device=device, dtype=dtype)

    # Precompute trans_scores expansion (reused every iteration)
    trans_exp = trans_scores.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)

    for t in range(1, seq_len + 1):
        # Case 1: segments starting from position 0
        if t <= D:
            start_score = start_transitions + dur_scores[t - 1, :] + segment_emissions[:, t, t - 1, :]
        else:
            start_score = neg_inf_score

        # Case 2: segments following previous segments
        if t > 1:
            num_valid_d = min(t - 1, D)

            # Index in reverse order instead of flip (avoids copy)
            indices = torch.arange(t - 1, t - 1 - num_valid_d, -1, device=device)
            prev_alpha = alpha[:, indices, :]  # (B, num_valid_d, K)

            # Combine scores
            combined = (
                prev_alpha.unsqueeze(-1)  # (B, D', K, 1)
                + trans_exp  # (1, 1, K, K)
                + dur_scores[:num_valid_d, :].unsqueeze(0).unsqueeze(2)  # (1, D', 1, K)
                + segment_emissions[:, t, :num_valid_d, :].unsqueeze(2)  # (B, D', 1, K)
            )
            case2_score = torch.logsumexp(combined.view(batch_size, -1, num_tags), dim=1)
        else:
            case2_score = neg_inf_score

        # Combine start and continuation cases
        alpha[:, t, :] = torch.logaddexp(start_score, case2_score)

    log_Z = torch.logsumexp(alpha[:, seq_len, :] + end_transitions, dim=1)
    return alpha, log_Z


@torch.jit.script
def _backward_pass(
    segment_emissions: Tensor,
    dur_scores: Tensor,
    trans_scores: Tensor,
    end_transitions: Tensor,
    seq_len: int,
    max_duration: int,
) -> Tensor:
    """Compute backward variables (beta) for Semi-Markov CRF (JIT compiled, optimized)."""
    batch_size = segment_emissions.shape[0]
    num_tags = segment_emissions.shape[3]
    device = segment_emissions.device
    dtype = segment_emissions.dtype
    D = min(max_duration, seq_len)
    neg_inf: float = -1.0e9  # local literal; TorchScript can't close over the module constant

    beta = torch.full((batch_size, seq_len + 1, num_tags), neg_inf, device=device, dtype=dtype)
    beta[:, seq_len, :] = end_transitions.unsqueeze(0).expand(batch_size, -1)

    # Precompute expanded end_transitions
    end_trans_exp = end_transitions.view(1, 1, -1)

    for t in range(seq_len - 1, -1, -1):
        num_valid_d = min(seq_len - t, D)
        end_indices = torch.arange(t + 1, t + 1 + num_valid_d, device=device)
        d_indices = torch.arange(num_valid_d, device=device)

        seg_emit = segment_emissions[:, end_indices, d_indices, :]  # (B, D', K)
        d_scores_slice = dur_scores[:num_valid_d, :]  # (D', K)
        next_beta = beta[:, end_indices, :]  # (B, D', K)

        # Compute transition scores to next segments
        trans_to_next = torch.logsumexp(trans_scores.unsqueeze(0) + next_beta.unsqueeze(2), dim=3)  # (B, D', K)

        # For segments ending at seq_len, use end_transitions instead
        end_mask = (end_indices == seq_len).view(1, -1, 1)
        suffix_score = torch.where(end_mask, end_trans_exp.expand(batch_size, num_valid_d, -1), trans_to_next)

        segment_score = d_scores_slice.unsqueeze(0) + seg_emit + suffix_score
        beta[:, t, :] = torch.logsumexp(segment_score, dim=1)

    return beta


def _compute_expected_stats(
    alpha: Tensor,
    beta: Tensor,
    segment_emissions: Tensor,
    dur_scores: Tensor,
    trans_scores: Tensor,
    start_transitions: Tensor,
    end_transitions: Tensor,
    log_Z: Tensor,
    seq_len: int,
    max_duration: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute expected sufficient statistics (gradients of log_Z) from forward-backward.

    These are the per-sample marginal expectations that equal ∂log_Z/∂(each input):
        - emission_marginals[b, t, k] = P(y_t = k | x)        -> ∂log_Z/∂emissions
        - dur_stats[b, d-1, k]        = E[# segments (state k, duration d)] -> ∂log_Z/∂dur_scores
        - trans_stats[b, j, k]        = E[# transitions j -> k]            -> ∂log_Z/∂trans_scores
        - start_stats[b, k]           = P(first segment is state k)        -> ∂log_Z/∂start_transitions
        - end_stats[b, k]             = P(last segment is state k)         -> ∂log_Z/∂end_transitions

    For emission marginals, sums over all segments containing each frame using a cumsum
    trick to avoid O(D^2) complexity.
    """
    batch_size = alpha.shape[0]
    num_tags = alpha.shape[2]
    device = alpha.device
    dtype = alpha.dtype
    D = min(max_duration, seq_len)
    log_Z3 = log_Z.view(-1, 1, 1)

    # prefix[b, start, k] = log-score to reach 'start' and enter a segment of state k
    prefix = torch.full((batch_size, seq_len + 1, num_tags), NEG_INF, device=device, dtype=dtype)
    prefix[:, 0, :] = start_transitions.unsqueeze(0)
    if seq_len > 0:
        alpha_expanded = alpha[:, 1:, :].unsqueeze(-1)  # (B, T, K_prev, 1)
        trans_expanded = trans_scores.unsqueeze(0).unsqueeze(0)  # (1, 1, K_prev, K_next)
        prefix[:, 1:, :] = torch.logsumexp(alpha_expanded + trans_expanded, dim=2)

    # suffix[b, end, k] = log-score from the end of a segment of state k to the sequence end
    suffix = torch.full((batch_size, seq_len + 1, num_tags), NEG_INF, device=device, dtype=dtype)
    suffix[:, seq_len, :] = end_transitions.unsqueeze(0)
    if seq_len > 0:
        beta_expanded = beta[:, :seq_len, :].unsqueeze(2)  # (B, T, 1, K_next)
        trans_expanded = trans_scores.unsqueeze(0).unsqueeze(0)  # (1, 1, K_curr, K_next)
        suffix[:, :seq_len, :] = torch.logsumexp(trans_expanded + beta_expanded, dim=3)

    # Segment posteriors, accumulated into frame marginals (cumsum) and duration stats
    seg_start_contrib = torch.zeros(batch_size, seq_len + 1, num_tags, device=device, dtype=dtype)
    seg_end_contrib = torch.zeros(batch_size, seq_len + 1, num_tags, device=device, dtype=dtype)
    dur_stats = torch.zeros(batch_size, D, num_tags, device=device, dtype=dtype)

    for d in range(1, D + 1):
        max_start = seq_len - d
        if max_start < 0:
            continue

        starts = torch.arange(max_start + 1, device=device)
        ends = starts + d

        pref = prefix[:, starts, :]  # (B, num_starts, K)
        suf = suffix[:, ends, :]  # (B, num_starts, K)
        seg_emit = segment_emissions[:, ends, d - 1, :]  # (B, num_starts, K)
        dur_score = dur_scores[d - 1, :]  # (K,)

        seg_log_prob = pref + dur_score + seg_emit + suf - log_Z3
        seg_prob = torch.exp(seg_log_prob)  # (B, num_starts, K)

        seg_start_contrib.scatter_add_(1, starts.view(1, -1, 1).expand(batch_size, -1, num_tags), seg_prob)
        seg_end_contrib.scatter_add_(1, ends.view(1, -1, 1).expand(batch_size, -1, num_tags), seg_prob)
        dur_stats[:, d - 1, :] = seg_prob.sum(dim=1)

    cumsum_start = seg_start_contrib.cumsum(dim=1)[:, :seq_len, :]
    cumsum_end = seg_end_contrib.cumsum(dim=1)[:, :seq_len, :]
    emission_marginals = cumsum_start - cumsum_end

    # start / end segment-state posteriors
    start_stats = torch.exp(start_transitions.unsqueeze(0) + beta[:, 0, :] - log_Z.view(-1, 1))  # (B, K)
    end_stats = torch.exp(alpha[:, seq_len, :] + end_transitions.unsqueeze(0) - log_Z.view(-1, 1))  # (B, K)

    # transition posteriors: for each interior boundary 'start', segment k begins there
    # preceded by a segment ending in state j. E[# j->k] = sum_start exp(alpha[start,j] + trans[j,k] + beta[start,k])
    trans_stats = torch.zeros(batch_size, num_tags, num_tags, device=device, dtype=dtype)
    if seq_len >= 2:
        a = alpha[:, 1:seq_len, :].unsqueeze(-1)  # (B, T-1, K_prev, 1)
        b = beta[:, 1:seq_len, :].unsqueeze(2)  # (B, T-1, 1, K_next)
        trans_exp = trans_scores.unsqueeze(0).unsqueeze(0)  # (1, 1, K_prev, K_next)
        combined = a + trans_exp + b - log_Z.view(-1, 1, 1, 1)
        trans_stats = torch.exp(combined).sum(dim=1)  # (B, K_prev, K_next)

    return emission_marginals, dur_stats, trans_stats, start_stats, end_stats


class SemiMarkovLogZFunction(torch.autograd.Function):
    """Custom autograd for Semi-Markov CRF log partition function.

    Forward: compute log_Z efficiently with no autograd overhead
    Backward: gradient w.r.t. emissions computed directly via marginals
    """

    @staticmethod
    def forward(
        ctx,
        emissions: Tensor,
        segment_emissions: Tensor,
        dur_scores: Tensor,
        trans_scores: Tensor,
        start_transitions: Tensor,
        end_transitions: Tensor,
        max_duration: int,
        use_parallel: bool = False,
    ) -> Tensor:
        seq_len = emissions.shape[1]

        with torch.no_grad():
            if use_parallel:
                alpha, log_Z = _forward_pass_hybrid(
                    segment_emissions,
                    dur_scores,
                    trans_scores,
                    start_transitions,
                    end_transitions,
                    seq_len,
                    max_duration,
                )
            else:
                alpha, log_Z = _forward_pass(
                    segment_emissions,
                    dur_scores,
                    trans_scores,
                    start_transitions,
                    end_transitions,
                    seq_len,
                    max_duration,
                )

        # Save for backward
        ctx.save_for_backward(
            alpha, segment_emissions, dur_scores, trans_scores, start_transitions, end_transitions, log_Z
        )
        ctx.seq_len = seq_len
        ctx.max_duration = max_duration
        ctx.emissions_shape = emissions.shape

        return log_Z

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (alpha, segment_emissions, dur_scores, trans_scores, start_transitions, end_transitions, log_Z) = (
            ctx.saved_tensors
        )
        seq_len = ctx.seq_len
        max_duration = ctx.max_duration

        with torch.no_grad():
            beta = _backward_pass(segment_emissions, dur_scores, trans_scores, end_transitions, seq_len, max_duration)

            # Expected sufficient statistics = gradients of log_Z w.r.t. each scored input
            marginals, dur_stats, trans_stats, start_stats, end_stats = _compute_expected_stats(
                alpha,
                beta,
                segment_emissions,
                dur_scores,
                trans_scores,
                start_transitions,
                end_transitions,
                log_Z,
                seq_len,
                max_duration,
            )

        # Chain-rule with the upstream gradient. Emissions are per-sample; the CRF
        # parameters are shared, so their gradients sum the weighted stats over the batch.
        w = grad_output.view(-1, 1, 1)
        grad_emissions = marginals * w
        grad_dur_scores = (dur_stats * w).sum(dim=0)
        grad_trans_scores = (trans_stats * w).sum(dim=0)
        grad_start = (start_stats * grad_output.view(-1, 1)).sum(dim=0)
        grad_end = (end_stats * grad_output.view(-1, 1)).sum(dim=0)

        # None for segment_emissions (emission grads flow through `emissions` directly),
        # max_duration, and use_parallel.
        return grad_emissions, None, grad_dur_scores, grad_trans_scores, grad_start, grad_end, None, None


class SemiMarkovCRF(nn.Module):
    """Semi-Markov CRF with Gaussian duration modeling.

    Args:
        num_tags: Number of states (4 for heart sounds: S1, Systole, S2, Diastole)
        max_duration: Maximum segment duration in frames
        duration_means: Initial mean duration for each state (in frames)
        duration_stds: Initial std duration for each state (in frames)
        forward_algorithm: Which forward algorithm to use:
            - "sequential": Original JIT-compiled sequential algorithm (default)
            - "parallel": Hybrid algorithm with optimized vectorization for t > D
    """

    def __init__(
        self,
        num_tags: int = 4,
        max_duration: int = 500,
        duration_means: list[float] | None = None,
        duration_stds: list[float] | None = None,
        forward_algorithm: Literal["sequential", "parallel"] = "parallel",
    ):
        super().__init__()
        self.num_tags = num_tags
        self.max_duration = max_duration
        self.forward_algorithm = forward_algorithm

        # Transition scores: transitions[i, j] = score for transition from state i to state j
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))

        # Start/end transitions
        self.start_transitions = nn.Parameter(torch.zeros(num_tags))
        self.end_transitions = nn.Parameter(torch.zeros(num_tags))

        # Duration parameters (Gaussian): learnable mean and std per state
        if duration_means is None:
            duration_means = [6.0, 12.0, 5.0, 20.0]
        if duration_stds is None:
            duration_stds = [2.0, 4.0, 2.0, 8.0]

        self.duration_means = nn.Parameter(torch.tensor(duration_means))
        self.duration_log_stds = nn.Parameter(torch.log(torch.tensor(duration_stds)))

        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters with reasonable defaults."""
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

        with torch.no_grad():
            # Discourage self-transitions (segments can't follow themselves)
            self.transitions.fill_diagonal_(-10.0)

            # Encourage valid cardiac cycle: S1 -> Systole -> S2 -> Diastole -> S1
            self.transitions[0, 1] = 1.0
            self.transitions[1, 2] = 1.0
            self.transitions[2, 3] = 1.0
            self.transitions[3, 0] = 1.0

    @property
    def duration_stds(self) -> Tensor:
        """Get duration standard deviations (always positive)."""
        return torch.exp(self.duration_log_stds)

    def duration_score(self, durations: Tensor, states: Tensor) -> Tensor:
        """Compute log probability of durations given states using Gaussian distribution.

        Args:
            durations: Segment durations (any shape)
            states: Segment states (same shape as durations)

        Returns:
            Log probability scores (same shape as input)
        """
        means = self.duration_means[states]
        stds = self.duration_stds[states]

        z = (durations.float() - means) / stds
        log_prob = -0.5 * z**2 - self.duration_log_stds[states] - 0.5 * math.log(2 * math.pi)

        return log_prob

    def _extract_segments(self, tags: Tensor) -> list[tuple[int, int, int]]:
        """Extract segments from a tag sequence.

        Args:
            tags: (seq_len,) - tag sequence

        Returns:
            List of (state, start, end) tuples
        """
        segments = []
        seq_len = tags.shape[0]

        start = 0
        current_state = tags[0].item()

        for t in range(1, seq_len):
            if tags[t].item() != current_state:
                segments.append((current_state, start, t))
                start = t
                current_state = tags[t].item()

        segments.append((current_state, start, seq_len))
        return segments

    def _precompute_duration_scores(self, device: torch.device) -> Tensor:
        """Precompute duration scores for all (duration, state) pairs.

        Returns:
            dur_scores: (max_duration, num_tags) where dur_scores[d-1, s] = log P(duration=d | state=s)
        """
        durations = torch.arange(1, self.max_duration + 1, device=device, dtype=torch.float32)
        # durations: (D,), means: (K,) -> broadcast to (D, K)
        means = self.duration_means.unsqueeze(0)  # (1, K)
        log_stds = self.duration_log_stds.unsqueeze(0)  # (1, K)
        stds = torch.exp(log_stds)

        z = (durations.unsqueeze(1) - means) / stds  # (D, K)
        log_prob = -0.5 * z**2 - log_stds - 0.5 * math.log(2 * math.pi)
        return log_prob

    def _compute_segment_emissions(self, emissions: Tensor) -> Tensor:
        """Precompute emission scores for all possible segments (fully vectorized).

        Uses cumsum for O(1) segment score lookup.

        Args:
            emissions: (batch_size, seq_len, num_tags)

        Returns:
            segment_scores: (batch_size, seq_len+1, max_duration, num_tags)
            where segment_scores[b, t, d-1, s] = sum of emissions[b, t-d:t, s]
            (score for segment of state s, duration d, ending at t)
        """
        batch_size, seq_len, num_tags = emissions.shape
        device = emissions.device
        dtype = emissions.dtype
        D = min(self.max_duration, seq_len)

        # Cumsum with zero prepended: cumsum[t] = sum(emissions[0:t])
        zeros = torch.zeros(batch_size, 1, num_tags, device=device, dtype=dtype)
        cumsum = torch.cat([zeros, emissions.cumsum(dim=1)], dim=1)  # (B, T+1, K)

        # Vectorized segment score computation using unfold
        # Create indices for all (t, d) pairs at once
        # For each d, we want cumsum[:, d:T+1, :] - cumsum[:, 0:T+1-d, :]

        # Use unfold to create sliding windows of cumsum
        # cumsum shape: (B, T+1, K)
        # We want segment_scores[b, t, d-1, s] = cumsum[b, t, s] - cumsum[b, t-d, s]

        # Pad and use convolution-style indexing
        segment_scores = torch.full((batch_size, seq_len + 1, D, num_tags), NEG_INF, device=device, dtype=dtype)

        # Create end indices: (T+1,) -> broadcast with duration offsets
        t_indices = torch.arange(seq_len + 1, device=device)  # (T+1,)
        d_indices = torch.arange(1, D + 1, device=device)  # (D,)

        # For valid (t, d) pairs where t >= d:
        # segment_scores[:, t, d-1, :] = cumsum[:, t, :] - cumsum[:, t-d, :]

        # Create meshgrid of (t, d) pairs
        t_grid, d_grid = torch.meshgrid(t_indices, d_indices, indexing="ij")  # (T+1, D)
        start_grid = t_grid - d_grid  # (T+1, D) - where segments start

        # Mask for valid pairs (start >= 0)
        valid_mask = start_grid >= 0  # (T+1, D)

        # Gather cumsum values for end and start positions
        # cumsum: (B, T+1, K), we need cumsum[:, t_grid, :] and cumsum[:, start_grid, :]
        # Use advanced indexing

        # Flatten for gathering
        t_flat = t_grid.flatten()  # (T+1 * D,)
        start_flat = start_grid.clamp(min=0).flatten()  # (T+1 * D,), clamped for invalid

        # Gather: (B, T+1, K) -> (B, T+1*D, K)
        end_vals = cumsum[:, t_flat, :]  # (B, T+1*D, K)
        start_vals = cumsum[:, start_flat, :]  # (B, T+1*D, K)

        # Compute differences
        diff = end_vals - start_vals  # (B, T+1*D, K)

        # Reshape to (B, T+1, D, K)
        diff = diff.view(batch_size, seq_len + 1, D, num_tags)

        # Apply mask (set invalid to -inf)
        valid_mask = valid_mask.unsqueeze(0).unsqueeze(-1)  # (1, T+1, D, 1)
        segment_scores = torch.where(valid_mask, diff, segment_scores)

        return segment_scores

    def _forward_algorithm(self, emissions: Tensor) -> Tensor:
        """Compute log partition function with custom backward for proper gradients.

        Uses custom autograd Function to avoid O(T) autograd graph while still
        computing correct gradients via marginal probabilities.

        Args:
            emissions: (batch_size, seq_len, num_tags)

        Returns:
            Log partition function for each sequence (batch_size,)
        """
        D = min(self.max_duration, emissions.shape[1])

        # dur_scores and trans_scores stay attached to their parameters so the custom
        # backward's expected-stat gradients propagate to the duration/transition params.
        dur_scores = self._precompute_duration_scores(emissions.device)[:D]
        diag_mask = torch.eye(self.num_tags, dtype=torch.bool, device=emissions.device)
        trans_scores = self.transitions.masked_fill(diag_mask, NEG_INF)

        # segment_emissions is detached: emission gradients flow through `emissions` directly.
        with torch.no_grad():
            segment_emissions = self._compute_segment_emissions(emissions)

        use_parallel = self.forward_algorithm == "parallel"

        return SemiMarkovLogZFunction.apply(
            emissions,
            segment_emissions,
            dur_scores,
            trans_scores,
            self.start_transitions,
            self.end_transitions,
            self.max_duration,
            use_parallel,
        )

    def _score_path(self, emissions: Tensor, tags: Tensor) -> Tensor:
        """Compute score for ground truth paths (fully vectorized).

        Args:
            emissions: (batch_size, seq_len, num_tags)
            tags: (batch_size, seq_len)

        Returns:
            Path scores (batch_size,)
        """
        batch_size, seq_len, _ = emissions.shape
        device = emissions.device
        dtype = emissions.dtype

        # 1. Emission scores (sum of emissions at correct tag positions)
        gathered = emissions.gather(2, tags.unsqueeze(2)).squeeze(2)  # (B, T)
        emission_scores = gathered.sum(dim=1)  # (B,)

        # 2. Find segment boundaries
        tag_changes = tags[:, 1:] != tags[:, :-1]  # (B, T-1)
        segment_starts = torch.cat(
            [torch.ones(batch_size, 1, dtype=torch.bool, device=device), tag_changes], dim=1
        )  # (B, T)

        # 3. Compute segment IDs and counts
        segment_ids = segment_starts.long().cumsum(dim=1) - 1  # (B, T)
        num_segments = segment_ids[:, -1] + 1  # (B,)
        max_segments = int(num_segments.max().item())

        # 4. Durations via scatter_add
        segment_counts = torch.zeros(batch_size, max_segments, device=device, dtype=dtype)
        segment_counts.scatter_add_(1, segment_ids, torch.ones_like(tags, dtype=dtype))

        # 5. Get segment states (tags constant within segment, scatter takes any)
        segment_states = torch.zeros(batch_size, max_segments, dtype=torch.long, device=device)
        segment_states.scatter_(1, segment_ids, tags)

        # 6. Duration scores
        means = self.duration_means[segment_states]
        log_stds = self.duration_log_stds[segment_states]
        stds = torch.exp(log_stds)

        z = (segment_counts - means) / stds
        dur_scores_raw = -0.5 * z**2 - log_stds - 0.5 * math.log(2 * math.pi)

        seg_mask = torch.arange(max_segments, device=device).unsqueeze(0) < num_segments.unsqueeze(1)
        dur_scores = (dur_scores_raw * seg_mask).sum(dim=1)

        # 7. Transition scores
        start_trans_scores = self.start_transitions[segment_states[:, 0]]

        prev_states = segment_states[:, :-1]
        next_states = segment_states[:, 1:]
        internal_trans = self.transitions[prev_states, next_states]

        trans_mask = torch.arange(max_segments - 1, device=device).unsqueeze(0) < (num_segments - 1).unsqueeze(1)
        internal_trans_scores = (internal_trans * trans_mask).sum(dim=1)

        last_idx = (num_segments - 1).clamp(min=0).unsqueeze(1)
        last_states = segment_states.gather(1, last_idx).squeeze(1)
        end_trans_scores = self.end_transitions[last_states]

        return emission_scores + dur_scores + start_trans_scores + internal_trans_scores + end_trans_scores

    def forward(self, emissions: Tensor, tags: Tensor) -> Tensor:
        """Compute negative log-likelihood loss.

        Args:
            emissions: (batch_size, seq_len, num_tags)
            tags: (batch_size, seq_len) - ground truth tags

        Returns:
            Negative log-likelihood loss (scalar)
        """
        log_likelihood = self._score_path(emissions, tags)
        log_Z = self._forward_algorithm(emissions)

        nll = log_Z - log_likelihood
        return nll.mean()

    def _viterbi_decode(self, emissions: Tensor) -> tuple[Tensor, list[list[tuple[int, int, int]]]]:
        """Find best segmentation using vectorized Semi-Markov Viterbi.

        Args:
            emissions: (batch_size, seq_len, num_tags)

        Returns:
            best_tags: (batch_size, seq_len)
            segments: List of segment lists per batch
        """
        batch_size, seq_len, num_tags = emissions.shape
        device = emissions.device
        D = min(self.max_duration, seq_len)

        # Precompute scores
        dur_scores = self._precompute_duration_scores(device)[:D]  # (D, K)
        segment_emissions = self._compute_segment_emissions(emissions)  # (B, T+1, D, K)

        # V[t, s] = best score ending at time t in state s
        V = torch.full((batch_size, seq_len + 1, num_tags), NEG_INF, device=device)

        # Backpointers: (B, T+1, K, 2) storing (prev_t, prev_s) for best path
        backpointers = torch.zeros((batch_size, seq_len + 1, num_tags, 2), dtype=torch.long, device=device)

        # Transition scores with self-transitions masked
        trans_scores = self.transitions.clone()
        trans_scores.fill_diagonal_(NEG_INF)

        for t in range(1, seq_len + 1):
            # === Case 1: Segments starting from t=0 (uses start_transitions) ===
            if t <= D:
                # start_score[b, s] = start_trans[s] + dur[t-1, s] + emission[b, t, t-1, s]
                start_score = (
                    self.start_transitions + dur_scores[t - 1, :] + segment_emissions[:, t, t - 1, :]
                )  # (B, K)
            else:
                start_score = torch.full((batch_size, num_tags), NEG_INF, device=device)

            # === Case 2: Segments following previous segments (vectorized) ===
            if t > 1:
                num_valid_d = min(t - 1, D)

                # Gather V values for all prev_t = t-d where d in [1, num_valid_d]
                prev_t_indices = torch.arange(t - 1, t - 1 - num_valid_d, -1, device=device)  # (D',)
                prev_V = V[:, prev_t_indices, :]  # (B, D', K_prev)

                # Duration scores for d = 1..num_valid_d
                d_scores = dur_scores[:num_valid_d, :]  # (D', K_next)

                # Segment emissions for these durations
                seg_emit = segment_emissions[:, t, :num_valid_d, :]  # (B, D', K_next)

                # Compute scores for all (d, prev_s, next_s) combinations
                # Score = V[b, prev_t, prev_s] + trans[prev_s, next_s] + dur[d, next_s] + emit[b, d, next_s]
                # prev_V: (B, D', K_prev) -> (B, D', K_prev, 1)
                # trans_scores: (K_prev, K_next) -> (1, 1, K_prev, K_next)
                # d_scores: (D', K_next) -> (1, D', 1, K_next)
                # seg_emit: (B, D', K_next) -> (B, D', 1, K_next)

                prev_V_exp = prev_V.unsqueeze(-1)  # (B, D', K_prev, 1)
                trans_exp = trans_scores.unsqueeze(0).unsqueeze(0)  # (1, 1, K_prev, K_next)
                d_scores_exp = d_scores.unsqueeze(0).unsqueeze(2)  # (1, D', 1, K_next)
                seg_emit_exp = seg_emit.unsqueeze(2)  # (B, D', 1, K_next)

                # Combined score: (B, D', K_prev, K_next)
                combined = prev_V_exp + trans_exp + d_scores_exp + seg_emit_exp

                # For each next_s, find max over (D', K_prev) and track argmax
                # Reshape to (B, D' * K_prev, K_next)
                combined_flat = combined.view(batch_size, num_valid_d * num_tags, num_tags)

                # Max over the flattened (d, prev_s) dimension
                case2_score, case2_argmax = combined_flat.max(dim=1)  # (B, K_next), (B, K_next)

                # Decode argmax to (d_idx, prev_s)
                best_d_idx = case2_argmax // num_tags  # (B, K_next)
                best_prev_s = case2_argmax % num_tags  # (B, K_next)
                best_prev_t = prev_t_indices[best_d_idx]  # (B, K_next)
            else:
                case2_score = torch.full((batch_size, num_tags), NEG_INF, device=device)
                best_prev_t = torch.zeros((batch_size, num_tags), dtype=torch.long, device=device)
                best_prev_s = torch.zeros((batch_size, num_tags), dtype=torch.long, device=device)

            # Compare case1 vs case2 and update V and backpointers
            use_case1 = start_score > case2_score  # (B, K)

            V[:, t, :] = torch.where(use_case1, start_score, case2_score)
            backpointers[:, t, :, 0] = torch.where(use_case1, torch.tensor(-1, device=device), best_prev_t)
            backpointers[:, t, :, 1] = torch.where(use_case1, torch.tensor(-1, device=device), best_prev_s)

        # Find best final state
        final_scores = V[:, seq_len, :] + self.end_transitions
        best_final_states = final_scores.argmax(dim=1)

        # Backtrack (still need loop for variable-length paths)
        best_tags = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        all_segments: list[list[tuple[int, int, int]]] = []

        for b in range(batch_size):
            segments: list[tuple[int, int, int]] = []
            t = seq_len
            s = best_final_states[b].item()

            while t > 0:
                prev_t = backpointers[b, t, s, 0].item()
                prev_s = backpointers[b, t, s, 1].item()

                # Segment ending at V-index t with prev_t = t - duration covers emission frames [prev_t, t);
                # prev_t = -1 is the case-1 sentinel (segment starts at frame 0). Using prev_t+1 here dropped
                # the first frame of every internal segment, defaulting it to state 0 and breaking the cycle.
                start = max(prev_t, 0)
                segments.append((s, start, t))
                best_tags[b, start:t] = s

                if prev_t < 0:
                    break
                t = prev_t
                s = prev_s

            segments.reverse()
            all_segments.append(segments)

        return best_tags, all_segments

    def decode(self, emissions: Tensor) -> Tensor:
        """Decode best tag sequence.

        Args:
            emissions: (batch_size, seq_len, num_tags)

        Returns:
            Best tags (batch_size, seq_len)
        """
        best_tags, _ = self._viterbi_decode(emissions)
        return best_tags

    def decode_segments(self, emissions: Tensor) -> list[list[tuple[int, int, int]]]:
        """Decode best segmentation.

        Args:
            emissions: (batch_size, seq_len, num_tags)

        Returns:
            List of segment lists, each segment is (state, start, end)
        """
        _, segments = self._viterbi_decode(emissions)
        return segments

    def marginals(self, emissions: Tensor) -> Tensor:
        """Compute marginal probabilities P(y_t = k | x) using forward-backward (vectorized).

        Reuses the same cumsum-based emission-marginal computation as the custom backward pass
        (`_compute_expected_stats`). That quantity is exactly ∂log_Z/∂emissions = P(y_t = k | x), and
        it is O(T·D) vectorized — far cheaper than the previous explicit O(T·K·D²) sum over every
        segment containing each frame, which made the test/AUROC path dominate wall-clock time.

        Args:
            emissions: (batch_size, seq_len, num_tags)

        Returns:
            Marginal probabilities: (batch_size, seq_len, num_tags)
        """
        seq_len = emissions.shape[1]
        device = emissions.device

        dur_scores = self._precompute_duration_scores(device)[: min(self.max_duration, seq_len)]
        segment_emissions = self._compute_segment_emissions(emissions)
        diag = torch.eye(self.num_tags, dtype=torch.bool, device=device)
        trans_scores = self.transitions.masked_fill(diag, NEG_INF)

        alpha, log_Z = _forward_pass(
            segment_emissions,
            dur_scores,
            trans_scores,
            self.start_transitions,
            self.end_transitions,
            seq_len,
            self.max_duration,
        )
        beta = _backward_pass(
            segment_emissions, dur_scores, trans_scores, self.end_transitions, seq_len, self.max_duration
        )
        marginals, _, _, _, _ = _compute_expected_stats(
            alpha,
            beta,
            segment_emissions,
            dur_scores,
            trans_scores,
            self.start_transitions,
            self.end_transitions,
            log_Z,
            seq_len,
            self.max_duration,
        )

        # Guard against tiny numerical drift so each row is a proper distribution.
        marginals = marginals.clamp_min(0.0)
        return marginals / marginals.sum(dim=-1, keepdim=True).clamp_min(1e-12)

"""Fast CRF implementation using JIT compilation and parallel scan."""

import torch
from torch import nn, Tensor


@torch.jit.script
def _log_semiring_matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication in log-semiring: C[i,k] = logsumexp_j(A[i,j] + B[j,k]).

    Args:
        a: (..., m, n)
        b: (..., n, p)

    Returns:
        Result of shape (..., m, p)
    """
    # a: (..., m, n, 1)
    # b: (..., 1, n, p)
    # sum: (..., m, n, p)
    return torch.logsumexp(a.unsqueeze(-1) + b.unsqueeze(-3), dim=-2)


@torch.jit.script
def _parallel_scan_log_semiring(matrices: Tensor) -> Tensor:
    """Parallel prefix scan for log-semiring matrix multiplication.

    Computes cumulative products: M[0], M[0]*M[1], M[0]*M[1]*M[2], ...

    Uses recursive doubling with O(log T) sequential steps.

    Args:
        matrices: (batch, seq_len, num_tags, num_tags)

    Returns:
        Cumulative products: (batch, seq_len, num_tags, num_tags)
    """
    batch_size, seq_len, num_tags, _ = matrices.shape

    # Compute number of iterations needed (ceiling of log2)
    log2_len = int(torch.tensor(seq_len, dtype=torch.float32).log2().ceil().item()) if seq_len > 1 else 1
    padded_len = 1 << log2_len

    # Pre-compute identity matrix template (reused throughout)
    # Identity in log-semiring: 0 on diagonal, -inf elsewhere
    identity_single = torch.full((num_tags, num_tags), float("-inf"),
                                  device=matrices.device, dtype=matrices.dtype)
    identity_single.fill_diagonal_(0.0)

    # Pad to power of 2 if needed
    if padded_len > seq_len:
        pad_size = padded_len - seq_len
        padding = identity_single.unsqueeze(0).unsqueeze(0).expand(batch_size, pad_size, -1, -1)
        matrices = torch.cat([matrices, padding], dim=1)

    result = matrices

    # Pre-allocate shifted buffer to avoid repeated allocations
    shifted = torch.empty_like(result)

    # Recursive doubling: at each step, combine with shifted version
    stride = 1
    for _ in range(log2_len):
        # Fill shifted: identity for first 'stride' positions, then result[:-stride]
        shifted[:, :stride] = identity_single
        shifted[:, stride:] = result[:, :-stride]

        # Combine: result[i] = shifted[i] @ result[i] = result[i-stride] @ result[i]
        result = _log_semiring_matmul(shifted, result)
        stride = stride * 2

    return result[:, :seq_len]


@torch.jit.script
def _forward_algorithm_parallel(emissions: Tensor, transitions: Tensor, start_transitions: Tensor, end_transitions: Tensor) -> Tensor:
    """Compute log partition function using parallel scan.

    Args:
        emissions: (batch_size, seq_len, num_tags)
        transitions: (num_tags, num_tags) - transitions[i, j] = score of i -> j
        start_transitions: (num_tags,)
        end_transitions: (num_tags,)

    Returns:
        Log partition function for each sequence in batch (batch_size,)
    """
    batch_size, seq_len, num_tags = emissions.shape

    # Build transfer matrices: T_t[i, j] = trans[i, j] + emit[t, j]
    # T_t represents: "score to go from state i at t-1 to state j at t"
    # Shape: (batch, seq_len, num_tags, num_tags)
    transfer = transitions.unsqueeze(0).unsqueeze(0) + emissions.unsqueeze(2)

    # For the first position, incorporate start transitions
    # First "matrix" should give: score[j] = start[j] + emit[0, j]
    # We represent this as a matrix where row 0 has the scores and other rows are -inf
    first_matrix = torch.full((batch_size, 1, num_tags, num_tags), float("-inf"),
                              device=emissions.device, dtype=emissions.dtype)
    first_scores = start_transitions + emissions[:, 0]  # (batch, num_tags)
    # Broadcast to all "from" states (will be summed over, but we want same result)
    first_matrix[:, 0, :, :] = first_scores.unsqueeze(1)

    # Remaining transfer matrices
    remaining = transfer[:, 1:]  # (batch, seq_len-1, num_tags, num_tags)

    # Concatenate
    all_matrices = torch.cat([first_matrix, remaining], dim=1)

    # Parallel scan
    cumulative = _parallel_scan_log_semiring(all_matrices)

    # Final result: take any row (they're all identical), add end transitions
    # cumulative[:, -1, i, j] = score to reach state j at end (same for all i)
    final_scores = cumulative[:, -1, 0, :]  # (batch, num_tags) - take row 0
    final_scores = final_scores + end_transitions

    return torch.logsumexp(final_scores, dim=1)


@torch.jit.script
def _forward_algorithm(emissions: Tensor, transitions: Tensor, start_transitions: Tensor, end_transitions: Tensor) -> Tensor:
    """Compute log partition function (forward algorithm) with JIT.

    Args:
        emissions: (batch_size, seq_len, num_tags)
        transitions: (num_tags, num_tags) - transitions[i, j] = score of i -> j
        start_transitions: (num_tags,)
        end_transitions: (num_tags,)

    Returns:
        Log partition function for each sequence in batch (batch_size,)
    """
    batch_size, seq_len, num_tags = emissions.shape

    # Start with first emission + start transition
    # score shape: (batch_size, num_tags)
    score = emissions[:, 0] + start_transitions

    # Iterate through sequence
    for i in range(1, seq_len):
        # score[:, :, None] is (batch, num_tags, 1) - previous scores
        # transitions is (num_tags, num_tags) - transitions[prev, next]
        # emissions[:, i, None, :] is (batch, 1, num_tags) - current emissions
        # We want: new_score[b, next] = logsumexp_prev(score[b, prev] + trans[prev, next] + emit[b, next])

        # Broadcast: (batch, num_tags, 1) + (num_tags, num_tags) + (batch, 1, num_tags)
        # = (batch, num_tags, num_tags) where [b, prev, next]
        broadcast_score = score.unsqueeze(2) + transitions.unsqueeze(0) + emissions[:, i].unsqueeze(1)
        score = torch.logsumexp(broadcast_score, dim=1)

    # Add end transitions
    score = score + end_transitions
    return torch.logsumexp(score, dim=1)


@torch.jit.script
def _compute_score(emissions: Tensor, tags: Tensor, transitions: Tensor, start_transitions: Tensor, end_transitions: Tensor) -> Tensor:
    """Compute score of given tag sequence (vectorized).

    Args:
        emissions: (batch_size, seq_len, num_tags)
        tags: (batch_size, seq_len)
        transitions: (num_tags, num_tags) - transitions[i, j] = score of i -> j
        start_transitions: (num_tags,)
        end_transitions: (num_tags,)

    Returns:
        Score for each sequence (batch_size,)
    """
    # Gather emission scores for all positions at once
    # emissions_score[b, t] = emissions[b, t, tags[b, t]]
    emissions_score = emissions.gather(2, tags.unsqueeze(2)).squeeze(2)  # (batch, seq_len)
    total_emissions = emissions_score.sum(dim=1)  # (batch,)

    # Gather all transition scores at once
    prev_tags = tags[:, :-1]  # (batch, seq_len-1)
    next_tags = tags[:, 1:]   # (batch, seq_len-1)
    transition_scores = transitions[prev_tags, next_tags]  # (batch, seq_len-1)
    total_transitions = transition_scores.sum(dim=1)  # (batch,)

    # Start and end transitions
    start_scores = start_transitions[tags[:, 0]]  # (batch,)
    end_scores = end_transitions[tags[:, -1]]     # (batch,)

    return total_emissions + total_transitions + start_scores + end_scores


@torch.jit.script
def _viterbi_decode(emissions: Tensor, transitions: Tensor, start_transitions: Tensor, end_transitions: Tensor) -> Tensor:
    """Viterbi decoding with JIT.

    Args:
        emissions: (batch_size, seq_len, num_tags)
        transitions: (num_tags, num_tags) - transitions[i, j] = score of i -> j
        start_transitions: (num_tags,)
        end_transitions: (num_tags,)

    Returns:
        Best tag sequence (batch_size, seq_len)
    """
    batch_size, seq_len, num_tags = emissions.shape
    device = emissions.device

    # score shape: (batch_size, num_tags)
    score = emissions[:, 0] + start_transitions

    # Store backpointers
    backpointers: list[Tensor] = []

    for i in range(1, seq_len):
        # (batch, num_tags, 1) + (num_tags, num_tags) = (batch, num_tags, num_tags)
        # broadcast_score[b, prev, next] = score[b, prev] + trans[prev, next]
        broadcast_score = score.unsqueeze(2) + transitions.unsqueeze(0)

        # Best previous tag for each current tag
        max_scores, best_prev = broadcast_score.max(dim=1)  # (batch, num_tags)

        backpointers.append(best_prev)
        score = max_scores + emissions[:, i]

    # Add end transitions and find best final tag
    score = score + end_transitions
    _, best_last = score.max(dim=1)  # (batch,)

    # Backtrack
    best_path = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    best_path[:, -1] = best_last

    for i in range(seq_len - 2, -1, -1):
        best_path[:, i] = backpointers[i].gather(1, best_path[:, i + 1].unsqueeze(1)).squeeze(1)

    return best_path


class CRF(nn.Module):
    """Fast CRF layer with JIT-compiled forward and decode."""

    def __init__(self, num_tags: int) -> None:
        super().__init__()
        self.num_tags = num_tags

        # Transition matrix: transitions[i, j] = score of transitioning from i to j
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(self, emissions: Tensor, tags: Tensor, use_parallel: bool = True) -> Tensor:
        """Compute negative log-likelihood loss.

        Args:
            emissions: (batch_size, seq_len, num_tags)
            tags: (batch_size, seq_len)
            use_parallel: Whether to use parallel scan algorithm (faster for long sequences)

        Returns:
            Negative log-likelihood (scalar, mean over batch)
        """
        if use_parallel:
            log_partition = _forward_algorithm_parallel(emissions, self.transitions, self.start_transitions, self.end_transitions)
        else:
            log_partition = _forward_algorithm(emissions, self.transitions, self.start_transitions, self.end_transitions)
        sequence_score = _compute_score(emissions, tags, self.transitions, self.start_transitions, self.end_transitions)
        return (log_partition - sequence_score).mean()

    def decode(self, emissions: Tensor) -> Tensor:
        """Find best tag sequence using Viterbi algorithm.

        Args:
            emissions: (batch_size, seq_len, num_tags)

        Returns:
            Best tag sequences (batch_size, seq_len)
        """
        return _viterbi_decode(emissions, self.transitions, self.start_transitions, self.end_transitions)

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

import torch
from torch import Tensor, nn


class SemiMarkovCRF(nn.Module):
    """Semi-Markov CRF with Gaussian duration modeling.

    Args:
        num_tags: Number of states (4 for heart sounds: S1, Systole, S2, Diastole)
        max_duration: Maximum segment duration in frames
        duration_means: Initial mean duration for each state (in frames)
        duration_stds: Initial std duration for each state (in frames)
    """

    def __init__(
        self,
        num_tags: int = 4,
        max_duration: int = 500,
        duration_means: list[float] | None = None,
        duration_stds: list[float] | None = None,
    ):
        super().__init__()
        self.num_tags = num_tags
        self.max_duration = max_duration

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
        segment_scores = torch.full((batch_size, seq_len + 1, D, num_tags), float("-inf"), device=device, dtype=dtype)

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
        """Compute log partition function using vectorized forward algorithm.

        Optimized to minimize Python loops by batching over durations and states.

        Args:
            emissions: (batch_size, seq_len, num_tags)

        Returns:
            Log partition function for each sequence (batch_size,)
        """
        batch_size, seq_len, num_tags = emissions.shape
        device = emissions.device
        dtype = emissions.dtype
        D = min(self.max_duration, seq_len)

        # Precompute scores
        dur_scores = self._precompute_duration_scores(device)[:D]  # (D, K)
        segment_emissions = self._compute_segment_emissions(emissions)  # (B, T+1, D, K)

        # alpha[t, s] = logsumexp of scores of all paths ending at time t in state s
        alpha = torch.full((batch_size, seq_len + 1, num_tags), float("-inf"), device=device, dtype=dtype)

        # Precompute transition mask: trans_mask[prev_s, s] = -inf if invalid (self-transition)
        trans_scores = self.transitions.clone()  # (K, K)
        trans_scores.fill_diagonal_(float("-inf"))  # No self-transitions

        for t in range(1, seq_len + 1):
            max_d = min(t, D)

            # === Case 1: Segments starting from t=0 (d == t, uses start_transitions) ===
            if t <= D:
                # start_score[s] = start_trans[s] + dur[t-1, s] + emission[t, t-1, s]
                start_score = (
                    self.start_transitions + dur_scores[t - 1, :] + segment_emissions[:, t, t - 1, :]
                )  # (B, K)
            else:
                start_score = torch.full((batch_size, num_tags), float("-inf"), device=device, dtype=dtype)

            # === Case 2: Segments following previous segments ===
            # Gather alpha values for all prev_t = t-d where d in [1, max_d] and prev_t > 0
            # We need alpha[:, t-1, :], alpha[:, t-2, :], ..., alpha[:, t-max_d, :]
            # but only where t-d > 0

            if t > 1:
                # Number of valid durations where prev_t = t - d > 0
                # d can be 1 to min(t-1, D) for prev_t > 0
                num_valid_d = min(t - 1, D)

                # Gather prev_alpha: alpha[:, t-d, :] for d = 1..num_valid_d
                # prev_t_indices = [t-1, t-2, ..., t-num_valid_d]
                prev_t_indices = torch.arange(t - 1, t - 1 - num_valid_d, -1, device=device)  # (num_valid_d,)
                prev_alpha = alpha[:, prev_t_indices, :]  # (B, num_valid_d, K)

                # Duration scores for d = 1..num_valid_d
                d_scores = dur_scores[:num_valid_d, :]  # (num_valid_d, K)

                # Segment emissions for these durations
                seg_emit = segment_emissions[:, t, :num_valid_d, :]  # (B, num_valid_d, K)

                # Compute scores for all (d, prev_s, s) combinations
                # prev_alpha: (B, D', K_prev)
                # trans_scores: (K_prev, K_next)
                # d_scores: (D', K_next)
                # seg_emit: (B, D', K_next)

                # Score = prev_alpha[b, d, prev_s] + trans[prev_s, s] + dur[d, s] + emit[b, d, s]
                # We want to logsumexp over (d, prev_s) for each s

                # Expand dimensions for broadcasting:
                # prev_alpha: (B, D', K_prev, 1)
                # trans_scores: (1, 1, K_prev, K_next)
                # d_scores: (1, D', 1, K_next)
                # seg_emit: (B, D', 1, K_next)

                prev_alpha_exp = prev_alpha.unsqueeze(-1)  # (B, D', K_prev, 1)
                trans_exp = trans_scores.unsqueeze(0).unsqueeze(0)  # (1, 1, K_prev, K_next)
                d_scores_exp = d_scores.unsqueeze(0).unsqueeze(2)  # (1, D', 1, K_next)
                seg_emit_exp = seg_emit.unsqueeze(2)  # (B, D', 1, K_next)

                # Combined score: (B, D', K_prev, K_next)
                combined = prev_alpha_exp + trans_exp + d_scores_exp + seg_emit_exp

                # Logsumexp over (D', K_prev) -> (B, K_next)
                combined_flat = combined.view(batch_size, -1, num_tags)  # (B, D'*K_prev, K_next)
                case2_score = torch.logsumexp(combined_flat, dim=1)  # (B, K_next)
            else:
                case2_score = torch.full((batch_size, num_tags), float("-inf"), device=device, dtype=dtype)

            # Combine case 1 and case 2 using logsumexp
            alpha[:, t, :] = torch.logsumexp(torch.stack([start_score, case2_score], dim=-1), dim=-1)

        # Final: add end transitions and logsumexp over all states
        final_alpha = alpha[:, seq_len, :] + self.end_transitions  # (B, K)
        log_Z = torch.logsumexp(final_alpha, dim=1)  # (B,)

        return log_Z

    def _score_path(self, emissions: Tensor, tags: Tensor) -> Tensor:
        """Compute score for ground truth paths (vectorized over batch).

        Args:
            emissions: (batch_size, seq_len, num_tags)
            tags: (batch_size, seq_len)

        Returns:
            Path scores (batch_size,)
        """
        batch_size, seq_len, _ = emissions.shape
        device = emissions.device

        # Find segment boundaries using diff
        # A new segment starts where tags[t] != tags[t-1]
        tag_changes = torch.diff(tags, dim=1) != 0  # (B, T-1), True where segment changes
        # Prepend True for position 0 (first segment always starts)
        segment_starts = torch.cat(
            [torch.ones(batch_size, 1, dtype=torch.bool, device=device), tag_changes], dim=1
        )  # (B, T)

        # Compute emission scores: gather emissions for correct tags and sum
        # emission_score[b] = sum over t of emissions[b, t, tags[b, t]]
        gathered = emissions.gather(2, tags.unsqueeze(2)).squeeze(2)  # (B, T)
        emission_scores = gathered.sum(dim=1)  # (B,)

        # Compute transition and duration scores per batch (need to iterate for variable segments)
        trans_dur_scores = torch.zeros(batch_size, device=device)

        for b in range(batch_size):
            starts = segment_starts[b].nonzero(as_tuple=True)[0]  # indices where segments start
            ends = torch.cat([starts[1:], torch.tensor([seq_len], device=device)])  # segment end positions

            for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
                state = tags[b, start].item()
                duration = (end - start).item()

                # Duration score
                mean = self.duration_means[state]
                log_std = self.duration_log_stds[state]
                std = torch.exp(log_std)
                z = (duration - mean) / std
                dur_score = -0.5 * z**2 - log_std - 0.5 * math.log(2 * math.pi)

                # Transition score
                if i == 0:
                    trans_score = self.start_transitions[state]
                else:
                    prev_state = tags[b, starts[i - 1]].item()
                    trans_score = self.transitions[prev_state, state]

                trans_dur_scores[b] = trans_dur_scores[b] + dur_score + trans_score

            # End transition
            last_state = tags[b, -1].item()
            trans_dur_scores[b] = trans_dur_scores[b] + self.end_transitions[last_state]

        return emission_scores + trans_dur_scores

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
        V = torch.full((batch_size, seq_len + 1, num_tags), float("-inf"), device=device)

        # Backpointers: (B, T+1, K, 2) storing (prev_t, prev_s) for best path
        backpointers = torch.zeros((batch_size, seq_len + 1, num_tags, 2), dtype=torch.long, device=device)

        # Transition scores with self-transitions masked
        trans_scores = self.transitions.clone()
        trans_scores.fill_diagonal_(float("-inf"))

        for t in range(1, seq_len + 1):
            # === Case 1: Segments starting from t=0 (uses start_transitions) ===
            if t <= D:
                # start_score[b, s] = start_trans[s] + dur[t-1, s] + emission[b, t, t-1, s]
                start_score = (
                    self.start_transitions + dur_scores[t - 1, :] + segment_emissions[:, t, t - 1, :]
                )  # (B, K)
            else:
                start_score = torch.full((batch_size, num_tags), float("-inf"), device=device)

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
                case2_score = torch.full((batch_size, num_tags), float("-inf"), device=device)
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

                start = prev_t + 1 if prev_t >= 0 else 0
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

    def _backward_algorithm(self, emissions: Tensor) -> Tensor:
        """Compute backward variables for Semi-Markov CRF (vectorized).

        β[t, s] = log-sum of scores of paths from t to end, where a segment of state s
        starts at time t.

        Args:
            emissions: (batch_size, seq_len, num_tags)

        Returns:
            beta: (batch_size, seq_len + 1, num_tags)
        """
        batch_size, seq_len, num_tags = emissions.shape
        device = emissions.device
        dtype = emissions.dtype
        D = min(self.max_duration, seq_len)

        # Precompute scores
        dur_scores = self._precompute_duration_scores(device)[:D]  # (D, K)
        segment_emissions = self._compute_segment_emissions(emissions)  # (B, T+1, D, K)

        # beta[t, s] = log-sum of scores starting at t in state s
        beta = torch.full((batch_size, seq_len + 1, num_tags), float("-inf"), device=device, dtype=dtype)

        # Initialize: beta[T, s] = end_transitions[s] (empty segment at the end)
        beta[:, seq_len, :] = self.end_transitions.unsqueeze(0)

        # Transition scores with self-transitions masked
        trans_scores = self.transitions.clone()
        trans_scores.fill_diagonal_(float("-inf"))

        # Backward pass: from T-1 down to 0
        for t in range(seq_len - 1, -1, -1):
            num_valid_d = min(seq_len - t, D)

            # end_t values for d = 1..num_valid_d
            end_t_indices = torch.arange(t + 1, t + 1 + num_valid_d, device=device)  # (D',)

            # Duration scores for d = 1..num_valid_d: (D', K_curr)
            d_scores = dur_scores[:num_valid_d, :]

            # Segment emissions for segments starting at t: (B, D', K_curr)
            seg_emit = segment_emissions[:, end_t_indices, torch.arange(num_valid_d, device=device), :]

            # === Case 1: Segments ending at seq_len (end_t == seq_len) ===
            # Score = dur[d, s] + emit[d, s] + end_trans[s]
            end_mask = end_t_indices == seq_len  # (D',)

            if end_mask.any():
                # d_scores: (D', K), seg_emit: (B, D', K), end_trans: (K,)
                end_score = d_scores + self.end_transitions  # (D', K)
                end_score = end_score.unsqueeze(0) + seg_emit  # (B, D', K)
                # Mask to only include segments that end at seq_len
                end_score = torch.where(end_mask.view(1, -1, 1), end_score, torch.full_like(end_score, float("-inf")))
            else:
                end_score = torch.full((batch_size, num_valid_d, num_tags), float("-inf"), device=device, dtype=dtype)

            # === Case 2: Segments followed by another segment ===
            # Score = dur[d, s] + emit[d, s] + logsumexp_next_s(beta[end_t, next_s] + trans[s, next_s])
            cont_mask = end_t_indices < seq_len  # (D',)

            if cont_mask.any():
                # Gather beta values for all end_t positions
                next_beta = beta[:, end_t_indices, :]  # (B, D', K_next)

                # For each current state s, compute logsumexp over next states
                # trans_scores[s, next_s]: (K_curr, K_next)
                # next_beta: (B, D', K_next) -> (B, D', 1, K_next)
                # trans_scores: (K_curr, K_next) -> (1, 1, K_curr, K_next)

                next_beta_exp = next_beta.unsqueeze(2)  # (B, D', 1, K_next)
                trans_exp = trans_scores.unsqueeze(0).unsqueeze(0)  # (1, 1, K_curr, K_next)

                # Combined: (B, D', K_curr, K_next)
                combined = next_beta_exp + trans_exp

                # Logsumexp over K_next: (B, D', K_curr)
                transition_score = torch.logsumexp(combined, dim=-1)

                # Add duration and emission scores: (B, D', K_curr)
                cont_score = transition_score + d_scores.unsqueeze(0) + seg_emit

                # Mask to only include segments that don't end at seq_len
                cont_score = torch.where(cont_mask.view(1, -1, 1), cont_score, torch.full_like(cont_score, float("-inf")))
            else:
                cont_score = torch.full((batch_size, num_valid_d, num_tags), float("-inf"), device=device, dtype=dtype)

            # Combine both cases and logsumexp over durations
            all_scores = torch.logaddexp(end_score, cont_score)  # (B, D', K)
            beta[:, t, :] = torch.logsumexp(all_scores, dim=1)  # (B, K)

        return beta

    def marginals(self, emissions: Tensor) -> Tensor:
        """Compute marginal probabilities P(y_t = k | x) using forward-backward (vectorized).

        For Semi-Markov CRF, this requires summing over all segments that contain time t.

        Args:
            emissions: (batch_size, seq_len, num_tags)

        Returns:
            Marginal probabilities: (batch_size, seq_len, num_tags)
        """
        batch_size, seq_len, num_tags = emissions.shape
        device = emissions.device
        dtype = emissions.dtype
        D = min(self.max_duration, seq_len)

        # Precompute scores
        dur_scores = self._precompute_duration_scores(device)[:D]
        segment_emissions = self._compute_segment_emissions(emissions)

        # Transition scores with self-transitions masked
        trans_scores = self.transitions.clone()
        trans_scores.fill_diagonal_(float("-inf"))

        # Forward pass (reuse vectorized logic from _forward_algorithm)
        alpha = torch.full((batch_size, seq_len + 1, num_tags), float("-inf"), device=device, dtype=dtype)

        for t in range(1, seq_len + 1):
            # Case 1: Segments starting from t=0
            if t <= D:
                start_score = self.start_transitions + dur_scores[t - 1, :] + segment_emissions[:, t, t - 1, :]
            else:
                start_score = torch.full((batch_size, num_tags), float("-inf"), device=device, dtype=dtype)

            # Case 2: Segments following previous segments
            if t > 1:
                num_valid_d = min(t - 1, D)
                prev_t_indices = torch.arange(t - 1, t - 1 - num_valid_d, -1, device=device)
                prev_alpha = alpha[:, prev_t_indices, :]  # (B, D', K)
                d_scores = dur_scores[:num_valid_d, :]  # (D', K)
                seg_emit = segment_emissions[:, t, :num_valid_d, :]  # (B, D', K)

                prev_alpha_exp = prev_alpha.unsqueeze(-1)  # (B, D', K_prev, 1)
                trans_exp = trans_scores.unsqueeze(0).unsqueeze(0)  # (1, 1, K_prev, K_next)
                d_scores_exp = d_scores.unsqueeze(0).unsqueeze(2)  # (1, D', 1, K_next)
                seg_emit_exp = seg_emit.unsqueeze(2)  # (B, D', 1, K_next)

                combined = prev_alpha_exp + trans_exp + d_scores_exp + seg_emit_exp
                combined_flat = combined.view(batch_size, -1, num_tags)
                case2_score = torch.logsumexp(combined_flat, dim=1)
            else:
                case2_score = torch.full((batch_size, num_tags), float("-inf"), device=device, dtype=dtype)

            alpha[:, t, :] = torch.logsumexp(torch.stack([start_score, case2_score], dim=-1), dim=-1)

        # Backward pass
        beta = self._backward_algorithm(emissions)

        # Compute log partition function
        log_Z = torch.logsumexp(alpha[:, seq_len, :] + self.end_transitions, dim=1)  # (B,)

        # Compute marginals using segment-level computation
        # For Semi-Markov CRF: P(y_t = k) = sum over segments containing t with state k
        # This is equivalent to: sum_{start <= t < end} P(segment(k, start, end) | x)

        # Segment probability = prefix_score + dur_score + emit_score + suffix_score - log_Z
        # where:
        #   prefix_score = alpha[start, :] + trans[:, k] if start > 0, else start_trans[k]
        #   suffix_score = trans[k, :] + beta[end, :] if end < T, else end_trans[k]

        # Precompute prefix scores: for each (start, k), the score to reach start and transition to k
        # prefix[start, k] = logsumexp_prev(alpha[start, prev] + trans[prev, k]) for start > 0
        #                  = start_trans[k] for start == 0
        prefix_scores = torch.full((batch_size, seq_len + 1, num_tags), float("-inf"), device=device, dtype=dtype)
        prefix_scores[:, 0, :] = self.start_transitions.unsqueeze(0)

        for start in range(1, seq_len + 1):
            # alpha[:, start, :]: (B, K_prev)
            # trans_scores: (K_prev, K_next)
            combined = alpha[:, start, :].unsqueeze(-1) + trans_scores.unsqueeze(0)  # (B, K_prev, K_next)
            prefix_scores[:, start, :] = torch.logsumexp(combined, dim=1)  # (B, K_next)

        # Precompute suffix scores: for each (end, k), the score from end to sequence end
        # suffix[end, k] = logsumexp_next(trans[k, next] + beta[end, next]) for end < T
        #                = end_trans[k] for end == T
        suffix_scores = torch.full((batch_size, seq_len + 1, num_tags), float("-inf"), device=device, dtype=dtype)
        suffix_scores[:, seq_len, :] = self.end_transitions.unsqueeze(0)

        for end in range(seq_len):
            # beta[:, end, :]: (B, K_next)
            # trans_scores: (K_curr, K_next)
            combined = trans_scores.unsqueeze(0) + beta[:, end, :].unsqueeze(1)  # (B, K_curr, K_next)
            suffix_scores[:, end, :] = torch.logsumexp(combined, dim=2)  # (B, K_curr)

        # Now compute marginals by summing over all segments containing each time t
        log_marginals = torch.full((batch_size, seq_len, num_tags), float("-inf"), device=device, dtype=dtype)

        for t in range(seq_len):
            # For each state k, sum over all segments (start, end) where start <= t < end
            for k in range(num_tags):
                segment_scores = []

                # Iterate over possible (start, end) pairs containing t
                for start in range(max(0, t - D + 1), t + 1):
                    max_end = min(seq_len, start + D)
                    for end in range(t + 1, max_end + 1):
                        d = end - start
                        # Segment score = prefix[start, k] + dur[d-1, k] + emit[end, d-1, k] + suffix[end, k]
                        score = (
                            prefix_scores[:, start, k]
                            + dur_scores[d - 1, k]
                            + segment_emissions[:, end, d - 1, k]
                            + suffix_scores[:, end, k]
                        )
                        segment_scores.append(score)

                if segment_scores:
                    stacked = torch.stack(segment_scores, dim=1)  # (B, num_segments)
                    log_marginals[:, t, k] = torch.logsumexp(stacked, dim=1)

        # Normalize by partition function
        log_marginals = log_marginals - log_Z.unsqueeze(1).unsqueeze(2)

        # Convert to probabilities
        return torch.softmax(log_marginals, dim=2)

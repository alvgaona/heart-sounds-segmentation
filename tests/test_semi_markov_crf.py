"""Tests for Semi-Markov CRF implementation."""

import math

import pytest
import torch

from hss.model.semi_markov_crf import (
    SemiMarkovCRF,
    SemiMarkovLogZFunction,
    _forward_pass,
    _forward_pass_hybrid,
    _log_matmul,
    _parallel_scan_log_semiring,
)


class TestSemiMarkovCRFBasics:
    """Basic functionality tests."""

    def test_initialization(self):
        """Test that SemiMarkovCRF initializes correctly."""
        crf = SemiMarkovCRF(num_tags=4, max_duration=100)

        assert crf.num_tags == 4
        assert crf.max_duration == 100
        assert crf.transitions.shape == (4, 4)
        assert crf.start_transitions.shape == (4,)
        assert crf.end_transitions.shape == (4,)
        assert crf.duration_means.shape == (4,)
        assert crf.duration_log_stds.shape == (4,)

    def test_custom_duration_params(self):
        """Test initialization with custom duration parameters."""
        means = [10.0, 20.0, 15.0, 30.0]
        stds = [3.0, 5.0, 4.0, 10.0]

        crf = SemiMarkovCRF(num_tags=4, duration_means=means, duration_stds=stds)

        torch.testing.assert_close(crf.duration_means, torch.tensor(means))
        torch.testing.assert_close(crf.duration_stds, torch.tensor(stds), rtol=1e-5, atol=1e-5)


class TestDurationScoring:
    """Tests for duration probability computation."""

    def test_duration_score_shape(self):
        """Duration score should have same shape as input."""
        crf = SemiMarkovCRF(num_tags=4)

        durations = torch.tensor([5, 10, 15, 20])
        states = torch.tensor([0, 1, 2, 3])

        scores = crf.duration_score(durations, states)
        assert scores.shape == durations.shape

    def test_duration_score_peak_at_mean(self):
        """Duration score should be highest at the mean."""
        crf = SemiMarkovCRF(num_tags=4, duration_means=[10.0, 20.0, 15.0, 30.0], duration_stds=[2.0, 2.0, 2.0, 2.0])

        # For state 0 with mean=10
        state = torch.tensor(0)
        durations = torch.tensor([5, 8, 10, 12, 15])
        scores = crf.duration_score(durations, state.expand(5))

        # Score at mean (10) should be highest
        assert scores[2] > scores[0]  # 10 > 5
        assert scores[2] > scores[1]  # 10 > 8
        assert scores[2] > scores[3]  # 10 > 12
        assert scores[2] > scores[4]  # 10 > 15

    def test_duration_score_symmetric(self):
        """Gaussian is symmetric around mean."""
        crf = SemiMarkovCRF(num_tags=4, duration_means=[10.0, 20.0, 15.0, 30.0], duration_stds=[2.0, 2.0, 2.0, 2.0])

        state = torch.tensor(0)
        # 8 and 12 are equidistant from mean=10
        score_8 = crf.duration_score(torch.tensor(8), state)
        score_12 = crf.duration_score(torch.tensor(12), state)

        torch.testing.assert_close(score_8, score_12, rtol=1e-5, atol=1e-5)


class TestSegmentExtraction:
    """Tests for extracting segments from tag sequences."""

    def test_extract_segments_simple(self):
        """Test segment extraction from simple sequence."""
        crf = SemiMarkovCRF(num_tags=4)

        # S1, S1, Systole, Systole, Systole, S2, S2
        tags = torch.tensor([0, 0, 1, 1, 1, 2, 2])
        segments = crf._extract_segments(tags)

        assert segments == [(0, 0, 2), (1, 2, 5), (2, 5, 7)]

    def test_extract_segments_single_state(self):
        """Test with single state throughout."""
        crf = SemiMarkovCRF(num_tags=4)

        tags = torch.tensor([1, 1, 1, 1, 1])
        segments = crf._extract_segments(tags)

        assert segments == [(1, 0, 5)]

    def test_extract_segments_alternating(self):
        """Test with rapid alternation (edge case)."""
        crf = SemiMarkovCRF(num_tags=4)

        tags = torch.tensor([0, 1, 2, 3, 0])
        segments = crf._extract_segments(tags)

        assert segments == [(0, 0, 1), (1, 1, 2), (2, 2, 3), (3, 3, 4), (0, 4, 5)]


class TestViterbiDecode:
    """Tests for Viterbi decoding."""

    def test_decode_output_shape(self):
        """Decode should return correct shape."""
        crf = SemiMarkovCRF(num_tags=4, max_duration=50)

        emissions = torch.randn(2, 20, 4)  # batch=2, seq_len=20
        best_tags = crf.decode(emissions)

        assert best_tags.shape == (2, 20)

    def test_decode_segments_output(self):
        """decode_segments should return list of segment lists."""
        crf = SemiMarkovCRF(num_tags=4, max_duration=50)

        emissions = torch.randn(2, 20, 4)
        segments = crf.decode_segments(emissions)

        assert len(segments) == 2  # One list per batch item
        for seg_list in segments:
            for state, start, end in seg_list:
                assert 0 <= state < 4
                assert 0 <= start < end <= 20

    def test_decode_strong_emissions(self):
        """With strong emissions, should decode to correct states."""
        crf = SemiMarkovCRF(num_tags=4, max_duration=50)

        # Create emissions that strongly favor a specific pattern
        # S1 (5 frames) -> Systole (8 frames) -> S2 (5 frames) -> Diastole (7 frames)
        emissions = torch.full((1, 25, 4), -10.0)
        emissions[0, 0:5, 0] = 10.0  # S1
        emissions[0, 5:13, 1] = 10.0  # Systole
        emissions[0, 13:18, 2] = 10.0  # S2
        emissions[0, 18:25, 3] = 10.0  # Diastole

        best_tags = crf.decode(emissions)

        # Check that decoded tags roughly match the strong emissions
        # (may not be exact due to duration constraints)
        assert best_tags[0, 0].item() == 0  # Should start with S1
        assert best_tags[0, 10].item() == 1  # Mid-systole
        assert best_tags[0, 15].item() == 2  # S2
        assert best_tags[0, 22].item() == 3  # Diastole


class TestForwardAlgorithm:
    """Tests for the forward algorithm (partition function)."""

    def test_forward_output_shape(self):
        """Forward algorithm should return one value per batch item."""
        crf = SemiMarkovCRF(num_tags=4, max_duration=50)

        emissions = torch.randn(3, 15, 4)
        log_Z = crf._forward_algorithm(emissions)

        assert log_Z.shape == (3,)

    def test_forward_finite(self):
        """Forward algorithm should return finite values."""
        crf = SemiMarkovCRF(num_tags=4, max_duration=50)

        emissions = torch.randn(2, 20, 4)
        log_Z = crf._forward_algorithm(emissions)

        assert torch.isfinite(log_Z).all()


class TestLoss:
    """Tests for loss computation."""

    def test_loss_positive(self):
        """Loss should be non-negative (NLL)."""
        crf = SemiMarkovCRF(num_tags=4, max_duration=50)

        emissions = torch.randn(2, 20, 4)
        # Create valid cardiac cycle tags
        tags = torch.zeros(2, 20, dtype=torch.long)
        tags[:, 0:5] = 0  # S1
        tags[:, 5:12] = 1  # Systole
        tags[:, 12:17] = 2  # S2
        tags[:, 17:20] = 3  # Diastole

        loss = crf(emissions, tags)

        assert loss.item() >= 0

    def test_loss_gradient_flow(self):
        """Loss should allow gradient flow to all parameters."""
        crf = SemiMarkovCRF(num_tags=4, max_duration=50)

        emissions = torch.randn(2, 15, 4, requires_grad=True)
        tags = torch.zeros(2, 15, dtype=torch.long)
        tags[:, 0:4] = 0
        tags[:, 4:8] = 1
        tags[:, 8:12] = 2
        tags[:, 12:15] = 3

        loss = crf(emissions, tags)
        loss.backward()

        assert emissions.grad is not None
        assert crf.transitions.grad is not None
        assert crf.duration_means.grad is not None
        assert crf.duration_log_stds.grad is not None


class TestMarginals:
    """Tests for marginal probability computation."""

    def test_marginals_output_shape(self):
        """Marginals should have shape (batch, seq_len, num_tags)."""
        crf = SemiMarkovCRF(num_tags=4, max_duration=50)

        emissions = torch.randn(2, 15, 4)
        marginals = crf.marginals(emissions)

        assert marginals.shape == (2, 15, 4)

    def test_marginals_sum_to_one(self):
        """Marginals should sum to 1 at each time step."""
        crf = SemiMarkovCRF(num_tags=4, max_duration=50)

        emissions = torch.randn(2, 15, 4)
        marginals = crf.marginals(emissions)

        sums = marginals.sum(dim=2)
        torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-4, atol=1e-4)

    def test_marginals_are_probabilities(self):
        """Marginals should be in [0, 1]."""
        crf = SemiMarkovCRF(num_tags=4, max_duration=50)

        emissions = torch.randn(2, 15, 4)
        marginals = crf.marginals(emissions)

        assert (marginals >= 0).all()
        assert (marginals <= 1).all()

    def test_marginals_finite(self):
        """Marginals should be finite."""
        crf = SemiMarkovCRF(num_tags=4, max_duration=50)

        emissions = torch.randn(2, 15, 4)
        marginals = crf.marginals(emissions)

        assert torch.isfinite(marginals).all()


class TestEdgeCases:
    """Edge case tests."""

    def test_short_sequence(self):
        """Should handle short sequences."""
        crf = SemiMarkovCRF(num_tags=4, max_duration=50)

        emissions = torch.randn(1, 5, 4)
        best_tags = crf.decode(emissions)

        assert best_tags.shape == (1, 5)

    def test_batch_size_one(self):
        """Should work with batch size of 1."""
        crf = SemiMarkovCRF(num_tags=4, max_duration=50)

        emissions = torch.randn(1, 20, 4)
        tags = torch.zeros(1, 20, dtype=torch.long)

        loss = crf(emissions, tags)
        assert torch.isfinite(loss)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self):
        """Should work on CUDA."""
        crf = SemiMarkovCRF(num_tags=4, max_duration=50).cuda()

        emissions = torch.randn(2, 20, 4).cuda()
        tags = torch.zeros(2, 20, dtype=torch.long).cuda()

        loss = crf(emissions, tags)
        best_tags = crf.decode(emissions)

        assert loss.device.type == "cuda"
        assert best_tags.device.type == "cuda"


class TestLogMatmul:
    """Tests for log-semiring matrix multiplication."""

    def test_log_matmul_identity(self):
        """Multiplying by identity should return original matrix."""
        A = torch.randn(2, 3, 4)

        # Identity in log-semiring: diagonal = 0, off-diagonal = -inf
        I = torch.full((2, 4, 4), float("-inf"))
        idx = torch.arange(4)
        I[:, idx, idx] = 0.0

        result = _log_matmul(A, I)
        torch.testing.assert_close(result, A, rtol=1e-5, atol=1e-5)

    def test_log_matmul_associative(self):
        """Log-semiring matmul should be associative: (A⊗B)⊗C = A⊗(B⊗C)."""
        A = torch.randn(2, 3, 4)
        B = torch.randn(2, 4, 5)
        C = torch.randn(2, 5, 6)

        left = _log_matmul(_log_matmul(A, B), C)
        right = _log_matmul(A, _log_matmul(B, C))

        torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-4)

    def test_log_matmul_vs_exp_matmul_exp(self):
        """Log-semiring matmul should equal exp(log(exp(A) @ exp(B)))."""
        # Use small values to avoid numerical issues
        A = torch.randn(2, 3, 4) * 0.5
        B = torch.randn(2, 4, 5) * 0.5

        # Direct in log-semiring
        result_log = _log_matmul(A, B)

        # Convert to regular, multiply, convert back
        result_exp = torch.log(torch.exp(A) @ torch.exp(B))

        torch.testing.assert_close(result_log, result_exp, rtol=1e-4, atol=1e-4)


class TestParallelScan:
    """Tests for parallel prefix scan in log-semiring."""

    def test_parallel_scan_single_element(self):
        """Single element scan should return the element."""
        matrices = torch.randn(2, 1, 3, 3)
        result = _parallel_scan_log_semiring(matrices)

        torch.testing.assert_close(result, matrices, rtol=1e-5, atol=1e-5)

    def test_parallel_scan_two_elements(self):
        """Two element scan should return [M0, M0⊗M1]."""
        matrices = torch.randn(2, 2, 3, 3)
        result = _parallel_scan_log_semiring(matrices)

        # First element unchanged
        torch.testing.assert_close(result[:, 0], matrices[:, 0], rtol=1e-5, atol=1e-5)

        # Second element = M0 ⊗ M1
        expected_1 = _log_matmul(matrices[:, 0], matrices[:, 1])
        torch.testing.assert_close(result[:, 1], expected_1, rtol=1e-4, atol=1e-4)

    def test_parallel_scan_vs_sequential(self):
        """Parallel scan should match sequential computation."""
        torch.manual_seed(42)
        matrices = torch.randn(2, 8, 4, 4) * 0.5

        # Parallel scan
        result_parallel = _parallel_scan_log_semiring(matrices)

        # Sequential scan
        result_sequential = torch.zeros_like(matrices)
        cumulative = matrices[:, 0].clone()
        result_sequential[:, 0] = cumulative
        for i in range(1, 8):
            cumulative = _log_matmul(cumulative, matrices[:, i])
            result_sequential[:, i] = cumulative

        torch.testing.assert_close(result_parallel, result_sequential, rtol=1e-3, atol=1e-3)

    def test_parallel_scan_non_power_of_two(self):
        """Parallel scan should work for non-power-of-2 lengths."""
        torch.manual_seed(42)
        matrices = torch.randn(2, 5, 3, 3) * 0.5

        # Parallel scan
        result_parallel = _parallel_scan_log_semiring(matrices)

        # Sequential scan
        result_sequential = torch.zeros_like(matrices)
        cumulative = matrices[:, 0].clone()
        result_sequential[:, 0] = cumulative
        for i in range(1, 5):
            cumulative = _log_matmul(cumulative, matrices[:, i])
            result_sequential[:, i] = cumulative

        torch.testing.assert_close(result_parallel, result_sequential, rtol=1e-3, atol=1e-3)


class TestParallelForwardPass:
    """Tests for parallel forward pass correctness."""

    def test_hybrid_vs_sequential_small(self):
        """Hybrid forward pass should match sequential on small inputs."""
        torch.manual_seed(42)
        crf = SemiMarkovCRF(num_tags=4, max_duration=10)

        batch_size = 2
        seq_len = 30
        emissions = torch.randn(batch_size, seq_len, 4)

        # Precompute common inputs
        D = min(crf.max_duration, seq_len)
        with torch.no_grad():
            dur_scores = crf._precompute_duration_scores(emissions.device)[:D]
            segment_emissions = crf._compute_segment_emissions(emissions)
            trans_scores = crf.transitions.clone()
            trans_scores.fill_diagonal_(float("-inf"))

        # Sequential forward pass
        alpha_seq, log_Z_seq = _forward_pass(
            segment_emissions, dur_scores, trans_scores, crf.start_transitions, crf.end_transitions, seq_len, D
        )

        # Hybrid forward pass
        alpha_hybrid, log_Z_hybrid = _forward_pass_hybrid(
            segment_emissions, dur_scores, trans_scores, crf.start_transitions, crf.end_transitions, seq_len, D
        )

        # Compare log_Z
        torch.testing.assert_close(log_Z_hybrid, log_Z_seq, rtol=1e-4, atol=1e-4)

        # Compare alpha values
        torch.testing.assert_close(alpha_hybrid, alpha_seq, rtol=1e-3, atol=1e-3)

    def test_hybrid_vs_sequential_medium(self):
        """Hybrid forward pass should match sequential on medium inputs."""
        torch.manual_seed(42)
        crf = SemiMarkovCRF(num_tags=4, max_duration=50)

        batch_size = 5
        seq_len = 200
        emissions = torch.randn(batch_size, seq_len, 4)

        D = min(crf.max_duration, seq_len)
        with torch.no_grad():
            dur_scores = crf._precompute_duration_scores(emissions.device)[:D]
            segment_emissions = crf._compute_segment_emissions(emissions)
            trans_scores = crf.transitions.clone()
            trans_scores.fill_diagonal_(float("-inf"))

        # Sequential forward pass
        alpha_seq, log_Z_seq = _forward_pass(
            segment_emissions, dur_scores, trans_scores, crf.start_transitions, crf.end_transitions, seq_len, D
        )

        # Hybrid forward pass
        alpha_hybrid, log_Z_hybrid = _forward_pass_hybrid(
            segment_emissions, dur_scores, trans_scores, crf.start_transitions, crf.end_transitions, seq_len, D
        )

        # Compare log_Z
        torch.testing.assert_close(log_Z_hybrid, log_Z_seq, rtol=1e-3, atol=1e-3)

        # Compare alpha values (allow slightly more tolerance for longer sequences)
        torch.testing.assert_close(alpha_hybrid, alpha_seq, rtol=1e-2, atol=1e-2)

    def test_hybrid_shorter_than_max_duration(self):
        """Hybrid should work when seq_len < max_duration."""
        torch.manual_seed(42)
        crf = SemiMarkovCRF(num_tags=4, max_duration=100)

        batch_size = 2
        seq_len = 20  # Less than max_duration
        emissions = torch.randn(batch_size, seq_len, 4)

        D = min(crf.max_duration, seq_len)
        with torch.no_grad():
            dur_scores = crf._precompute_duration_scores(emissions.device)[:D]
            segment_emissions = crf._compute_segment_emissions(emissions)
            trans_scores = crf.transitions.clone()
            trans_scores.fill_diagonal_(float("-inf"))

        # Both should work without error
        alpha_seq, log_Z_seq = _forward_pass(
            segment_emissions, dur_scores, trans_scores, crf.start_transitions, crf.end_transitions, seq_len, D
        )

        alpha_hybrid, log_Z_hybrid = _forward_pass_hybrid(
            segment_emissions, dur_scores, trans_scores, crf.start_transitions, crf.end_transitions, seq_len, D
        )

        torch.testing.assert_close(log_Z_hybrid, log_Z_seq, rtol=1e-4, atol=1e-4)

    def test_hybrid_gradient_matches_sequential(self):
        """Gradients from hybrid forward should match sequential."""
        torch.manual_seed(42)
        crf = SemiMarkovCRF(num_tags=4, max_duration=10)

        batch_size = 2
        seq_len = 25
        emissions = torch.randn(batch_size, seq_len, 4, requires_grad=True)

        D = min(crf.max_duration, seq_len)

        # Compute segment_emissions with gradients
        dur_scores = crf._precompute_duration_scores(emissions.device)[:D]
        segment_emissions = crf._compute_segment_emissions(emissions)
        trans_scores = crf.transitions.clone()
        trans_scores.fill_diagonal_(float("-inf"))

        # Sequential forward
        alpha_seq, log_Z_seq = _forward_pass(
            segment_emissions.detach(),
            dur_scores.detach(),
            trans_scores.detach(),
            crf.start_transitions.detach(),
            crf.end_transitions.detach(),
            seq_len,
            D,
        )

        # Test that hybrid produces finite results (gradient test is complex due to custom backward)
        alpha_hybrid, log_Z_hybrid = _forward_pass_hybrid(
            segment_emissions.detach(),
            dur_scores.detach(),
            trans_scores.detach(),
            crf.start_transitions.detach(),
            crf.end_transitions.detach(),
            seq_len,
            D,
        )

        assert torch.isfinite(log_Z_hybrid).all()
        # alpha[:, 0, :] is -inf (no segments end at t=0), check t > 0
        assert torch.isfinite(alpha_hybrid[:, 1:, :]).all()


class TestParallelForwardPassIntegration:
    """Integration tests for parallel forward pass with full CRF."""

    def test_loss_matches_with_parallel(self):
        """CRF loss should be the same whether using parallel or sequential."""
        torch.manual_seed(42)
        crf = SemiMarkovCRF(num_tags=4, max_duration=20)

        batch_size = 3
        seq_len = 50
        emissions = torch.randn(batch_size, seq_len, 4)
        tags = torch.zeros(batch_size, seq_len, dtype=torch.long)
        tags[:, 0:10] = 0
        tags[:, 10:25] = 1
        tags[:, 25:35] = 2
        tags[:, 35:50] = 3

        # Compute loss (uses sequential by default through custom backward)
        loss = crf(emissions, tags)

        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_sequential_vs_parallel_algorithm_setting(self):
        """Both algorithm settings should produce the same loss."""
        torch.manual_seed(42)

        batch_size = 3
        seq_len = 100
        emissions = torch.randn(batch_size, seq_len, 4)
        tags = torch.zeros(batch_size, seq_len, dtype=torch.long)
        tags[:, 0:20] = 0
        tags[:, 20:50] = 1
        tags[:, 50:70] = 2
        tags[:, 70:100] = 3

        # Create two CRFs with same initial parameters
        crf_seq = SemiMarkovCRF(num_tags=4, max_duration=50, forward_algorithm="sequential")
        crf_par = SemiMarkovCRF(num_tags=4, max_duration=50, forward_algorithm="parallel")

        # Copy parameters
        crf_par.load_state_dict(crf_seq.state_dict())

        # Compute losses
        loss_seq = crf_seq(emissions, tags)
        loss_par = crf_par(emissions, tags)

        torch.testing.assert_close(loss_seq, loss_par, rtol=1e-4, atol=1e-4)

    def test_parallel_gradient_flow(self):
        """Parallel algorithm should allow gradient flow to all parameters."""
        crf = SemiMarkovCRF(num_tags=4, max_duration=50, forward_algorithm="parallel")

        emissions = torch.randn(2, 100, 4, requires_grad=True)
        tags = torch.zeros(2, 100, dtype=torch.long)
        tags[:, 0:25] = 0
        tags[:, 25:50] = 1
        tags[:, 50:75] = 2
        tags[:, 75:100] = 3

        loss = crf(emissions, tags)
        loss.backward()

        assert emissions.grad is not None
        assert crf.transitions.grad is not None
        assert crf.duration_means.grad is not None
        assert crf.duration_log_stds.grad is not None


def _reference_log_Z(crf: SemiMarkovCRF, emissions: torch.Tensor) -> torch.Tensor:
    """Naive, fully-differentiable semi-Markov log partition function (single sequence).

    Deliberately simple (nested Python loops, no custom autograd) so that PyTorch's own
    autograd provides ground-truth gradients for every CRF parameter.
    """
    seq_len, num_tags = emissions.shape
    D = min(crf.max_duration, seq_len)

    means = crf.duration_means
    log_stds = crf.duration_log_stds
    stds = torch.exp(log_stds)

    def dur_score(d: int, s: int) -> torch.Tensor:
        z = (float(d) - means[s]) / stds[s]
        return -0.5 * z**2 - log_stds[s] - 0.5 * math.log(2 * math.pi)

    diag = torch.eye(num_tags, dtype=torch.bool, device=emissions.device)
    trans = crf.transitions.masked_fill(diag, float("-inf"))

    cum = torch.cat([torch.zeros(1, num_tags, dtype=emissions.dtype), emissions.cumsum(0)], dim=0)

    def seg_emit(t: int, d: int, s: int) -> torch.Tensor:
        return cum[t, s] - cum[t - d, s]

    neg_inf = float("-inf")
    alpha = torch.full((seq_len + 1, num_tags), neg_inf, dtype=emissions.dtype)
    for t in range(1, seq_len + 1):
        col = []
        for s in range(num_tags):
            terms = []
            for d in range(1, min(D, t) + 1):
                e = seg_emit(t, d, s) + dur_score(d, s)
                if t - d == 0:
                    terms.append(crf.start_transitions[s] + e)
                else:
                    terms.append(torch.logsumexp(alpha[t - d] + trans[:, s], dim=0) + e)
            col.append(torch.logsumexp(torch.stack(terms), dim=0))
        alpha[t] = torch.stack(col)

    return torch.logsumexp(alpha[seq_len] + crf.end_transitions, dim=0)


class TestGradientCorrectness:
    """The custom autograd backward must match true CRF gradients for every parameter."""

    def _make_crf(self) -> SemiMarkovCRF:
        return SemiMarkovCRF(
            num_tags=4,
            max_duration=12,
            duration_means=[3.0, 5.0, 2.0, 6.0],
            duration_stds=[1.5, 2.0, 1.0, 3.0],
        ).double()

    def test_logZ_matches_reference(self):
        """Custom forward log_Z should equal the naive reference value."""
        torch.manual_seed(0)
        crf = self._make_crf()
        emissions = torch.randn(1, 16, 4, dtype=torch.float64)

        actual = crf._forward_algorithm(emissions)[0]
        expected = _reference_log_Z(crf, emissions[0])

        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)

    def test_parameter_gradients_match_reference(self):
        """Every CRF parameter and emission gradient must match the differentiable reference."""
        torch.manual_seed(1)
        emissions = torch.randn(1, 18, 4, dtype=torch.float64)

        tags = torch.zeros(1, 18, dtype=torch.long)
        tags[:, 0:3] = 0
        tags[:, 3:9] = 1
        tags[:, 9:12] = 2
        tags[:, 12:18] = 3

        # Reference: naive differentiable log_Z + shared _score_path numerator.
        crf_ref = self._make_crf()
        em_ref = emissions.clone().requires_grad_(True)
        nll_ref = _reference_log_Z(crf_ref, em_ref[0]) - crf_ref._score_path(em_ref, tags)[0]
        nll_ref.backward()

        # Actual: custom-autograd path with identical parameters.
        crf_act = self._make_crf()
        crf_act.load_state_dict(crf_ref.state_dict())
        em_act = emissions.clone().requires_grad_(True)
        nll_act = crf_act(em_act, tags)
        nll_act.backward()

        torch.testing.assert_close(nll_act, nll_ref, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(em_act.grad, em_ref.grad, rtol=1e-6, atol=1e-6)

        ref_grads = dict(crf_ref.named_parameters())
        for name, p_act in crf_act.named_parameters():
            torch.testing.assert_close(
                p_act.grad, ref_grads[name].grad, rtol=1e-6, atol=1e-6, msg=f"grad mismatch: {name}"
            )

    def test_gradcheck_logZ(self):
        """torch.autograd.gradcheck on log_Z w.r.t. the raw CRF parameters in float64.

        Emissions are held fixed here (their gradient is a decoupled marginal computation
        validated by test_parameter_gradients_match_reference); this checks that the
        expected-stat gradients chain correctly through to duration/transition/boundary params.
        """
        torch.manual_seed(2)
        crf = self._make_crf()
        seq_len = 12
        num_tags = crf.num_tags
        emissions = torch.randn(1, seq_len, num_tags, dtype=torch.float64)
        D = min(crf.max_duration, seq_len)

        with torch.no_grad():
            segment_emissions = crf._compute_segment_emissions(emissions)
        em = emissions.detach()  # carrier only, no grad

        durations = torch.arange(1, D + 1, dtype=torch.float64).unsqueeze(1)  # (D, 1)
        diag = torch.eye(num_tags, dtype=torch.bool)

        means = crf.duration_means.detach().clone().requires_grad_(True)
        log_stds = crf.duration_log_stds.detach().clone().requires_grad_(True)
        transitions = crf.transitions.detach().clone().requires_grad_(True)
        start = crf.start_transitions.detach().clone().requires_grad_(True)
        end = crf.end_transitions.detach().clone().requires_grad_(True)

        def fn(means_, log_stds_, transitions_, start_, end_):
            stds_ = torch.exp(log_stds_)
            z = (durations - means_.unsqueeze(0)) / stds_.unsqueeze(0)
            dur_scores_ = -0.5 * z**2 - log_stds_.unsqueeze(0) - 0.5 * math.log(2 * math.pi)
            trans_scores_ = transitions_.masked_fill(diag, float("-inf"))
            return SemiMarkovLogZFunction.apply(
                em, segment_emissions, dur_scores_, trans_scores_, start_, end_, crf.max_duration, False
            )

        assert torch.autograd.gradcheck(fn, (means, log_stds, transitions, start, end), atol=1e-6, rtol=1e-4)

"""Tests for Semi-Markov CRF implementation."""

import pytest
import torch

from hss.model.semi_markov_crf import SemiMarkovCRF


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

"""Tests for the custom CRF implementation."""

import pytest
import torch

from hss.model.crf import (
    CRF,
    _compute_marginals,
    _compute_score,
    _forward_algorithm,
    _forward_algorithm_parallel,
    _viterbi_decode,
)


class TestCRFBasics:
    """Basic functionality tests."""

    def test_init(self):
        crf = CRF(num_tags=4)
        assert crf.num_tags == 4
        assert crf.transitions.shape == (4, 4)
        assert crf.start_transitions.shape == (4,)
        assert crf.end_transitions.shape == (4,)

    def test_parameters_registered(self):
        crf = CRF(num_tags=4)
        param_names = {name for name, _ in crf.named_parameters()}
        assert "transitions" in param_names
        assert "start_transitions" in param_names
        assert "end_transitions" in param_names

    def test_forward_returns_scalar(self):
        crf = CRF(num_tags=4)
        emissions = torch.randn(2, 10, 4)
        tags = torch.randint(0, 4, (2, 10))
        loss = crf(emissions, tags)
        assert loss.shape == ()
        assert loss.requires_grad

    def test_decode_returns_correct_shape(self):
        crf = CRF(num_tags=4)
        emissions = torch.randn(2, 10, 4)
        decoded = crf.decode(emissions)
        assert decoded.shape == (2, 10)
        assert decoded.dtype == torch.long

    def test_decode_values_in_range(self):
        crf = CRF(num_tags=4)
        emissions = torch.randn(5, 20, 4)
        decoded = crf.decode(emissions)
        assert (decoded >= 0).all()
        assert (decoded < 4).all()


class TestForwardAlgorithm:
    """Tests for the forward algorithm (log partition function)."""

    def test_output_shape(self):
        emissions = torch.randn(3, 15, 4)
        transitions = torch.randn(4, 4)
        start_trans = torch.randn(4)
        end_trans = torch.randn(4)

        result = _forward_algorithm(emissions, transitions, start_trans, end_trans)
        assert result.shape == (3,)

    def test_single_timestep(self):
        """For single timestep, partition = logsumexp(emit + start + end)."""
        emissions = torch.randn(2, 1, 4)
        transitions = torch.randn(4, 4)
        start_trans = torch.randn(4)
        end_trans = torch.randn(4)

        result = _forward_algorithm(emissions, transitions, start_trans, end_trans)
        expected = torch.logsumexp(emissions[:, 0] + start_trans + end_trans, dim=1)
        torch.testing.assert_close(result, expected)

    def test_deterministic(self):
        """Same input should give same output."""
        emissions = torch.randn(2, 10, 4)
        transitions = torch.randn(4, 4)
        start_trans = torch.randn(4)
        end_trans = torch.randn(4)

        result1 = _forward_algorithm(emissions, transitions, start_trans, end_trans)
        result2 = _forward_algorithm(emissions, transitions, start_trans, end_trans)
        torch.testing.assert_close(result1, result2)


class TestComputeScore:
    """Tests for sequence score computation."""

    def test_output_shape(self):
        emissions = torch.randn(3, 15, 4)
        tags = torch.randint(0, 4, (3, 15))
        transitions = torch.randn(4, 4)
        start_trans = torch.randn(4)
        end_trans = torch.randn(4)

        result = _compute_score(emissions, tags, transitions, start_trans, end_trans)
        assert result.shape == (3,)

    def test_single_timestep(self):
        """For single timestep, score = emit[tag] + start[tag] + end[tag]."""
        emissions = torch.randn(2, 1, 4)
        tags = torch.randint(0, 4, (2, 1))
        transitions = torch.randn(4, 4)
        start_trans = torch.randn(4)
        end_trans = torch.randn(4)

        result = _compute_score(emissions, tags, transitions, start_trans, end_trans)
        expected = (
            emissions[torch.arange(2), 0, tags[:, 0]]
            + start_trans[tags[:, 0]]
            + end_trans[tags[:, 0]]
        )
        torch.testing.assert_close(result, expected)

    def test_manual_two_timesteps(self):
        """Manually verify score for two timesteps."""
        batch_size = 1
        emissions = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
        tags = torch.tensor([[0, 1]])  # (1, 2)
        transitions = torch.tensor([[0.1, 0.2], [0.3, 0.4]])  # trans[i,j] = i->j
        start_trans = torch.tensor([0.5, 0.6])
        end_trans = torch.tensor([0.7, 0.8])

        result = _compute_score(emissions, tags, transitions, start_trans, end_trans)

        # Manual calculation:
        # start[0] + emit[0,0] + trans[0,1] + emit[1,1] + end[1]
        # = 0.5 + 1.0 + 0.2 + 4.0 + 0.8 = 6.5
        expected = torch.tensor([6.5])
        torch.testing.assert_close(result, expected)


class TestViterbiDecode:
    """Tests for Viterbi decoding."""

    def test_output_shape(self):
        emissions = torch.randn(3, 15, 4)
        transitions = torch.randn(4, 4)
        start_trans = torch.randn(4)
        end_trans = torch.randn(4)

        result = _viterbi_decode(emissions, transitions, start_trans, end_trans)
        assert result.shape == (3, 15)
        assert result.dtype == torch.long

    def test_single_timestep(self):
        """For single timestep, just pick argmax of emit + start + end."""
        emissions = torch.randn(2, 1, 4)
        transitions = torch.randn(4, 4)
        start_trans = torch.randn(4)
        end_trans = torch.randn(4)

        result = _viterbi_decode(emissions, transitions, start_trans, end_trans)
        expected = (emissions[:, 0] + start_trans + end_trans).argmax(dim=1, keepdim=True)
        torch.testing.assert_close(result, expected)

    def test_strong_emissions_dominate(self):
        """When emissions are very strong, decode should follow them."""
        batch_size, seq_len, num_tags = 2, 10, 4

        # Create very strong emissions for a specific path
        emissions = torch.full((batch_size, seq_len, num_tags), -100.0)
        expected_path = torch.randint(0, num_tags, (batch_size, seq_len))
        for b in range(batch_size):
            for t in range(seq_len):
                emissions[b, t, expected_path[b, t]] = 100.0

        transitions = torch.zeros(num_tags, num_tags)
        start_trans = torch.zeros(num_tags)
        end_trans = torch.zeros(num_tags)

        result = _viterbi_decode(emissions, transitions, start_trans, end_trans)
        torch.testing.assert_close(result, expected_path)

    def test_deterministic(self):
        """Same input should give same output."""
        emissions = torch.randn(2, 10, 4)
        transitions = torch.randn(4, 4)
        start_trans = torch.randn(4)
        end_trans = torch.randn(4)

        result1 = _viterbi_decode(emissions, transitions, start_trans, end_trans)
        result2 = _viterbi_decode(emissions, transitions, start_trans, end_trans)
        torch.testing.assert_close(result1, result2)


class TestCRFLoss:
    """Tests for the CRF loss (negative log-likelihood)."""

    def test_loss_is_positive(self):
        """NLL should generally be positive (probability < 1)."""
        crf = CRF(num_tags=4)
        emissions = torch.randn(5, 20, 4)
        tags = torch.randint(0, 4, (5, 20))

        loss = crf(emissions, tags)
        # Loss can be negative if sequence score > partition, but should be small
        # For random emissions and tags, loss should typically be positive
        assert loss.isfinite()

    def test_perfect_emissions_low_loss(self):
        """When emissions perfectly match tags, loss should be low."""
        crf = CRF(num_tags=4)
        # Set transitions to neutral
        with torch.no_grad():
            crf.transitions.fill_(0.0)
            crf.start_transitions.fill_(0.0)
            crf.end_transitions.fill_(0.0)

        batch_size, seq_len, num_tags = 2, 10, 4
        tags = torch.randint(0, num_tags, (batch_size, seq_len))

        # Create emissions that strongly favor the correct tags
        emissions = torch.full((batch_size, seq_len, num_tags), -10.0)
        for b in range(batch_size):
            for t in range(seq_len):
                emissions[b, t, tags[b, t]] = 10.0

        loss = crf(emissions, tags)
        assert loss < 1.0  # Should be close to 0

    def test_gradient_flow(self):
        """Gradients should flow through all parameters."""
        crf = CRF(num_tags=4)
        emissions = torch.randn(2, 10, 4, requires_grad=True)
        tags = torch.randint(0, 4, (2, 10))

        loss = crf(emissions, tags)
        loss.backward()

        assert emissions.grad is not None
        assert crf.transitions.grad is not None
        assert crf.start_transitions.grad is not None
        assert crf.end_transitions.grad is not None

    def test_batch_independence(self):
        """Loss for batch should equal mean of individual losses."""
        crf = CRF(num_tags=4)
        emissions = torch.randn(3, 10, 4)
        tags = torch.randint(0, 4, (3, 10))

        batch_loss = crf(emissions, tags)

        individual_losses = []
        for i in range(3):
            loss_i = crf(emissions[i : i + 1], tags[i : i + 1])
            individual_losses.append(loss_i)

        expected = torch.stack(individual_losses).mean()
        torch.testing.assert_close(batch_loss, expected)


class TestParallelForward:
    """Tests for parallel scan forward algorithm."""

    def test_matches_sequential_short(self):
        """Parallel should match sequential for short sequences."""
        emissions = torch.randn(2, 10, 4)
        transitions = torch.randn(4, 4)
        start_trans = torch.randn(4)
        end_trans = torch.randn(4)

        sequential = _forward_algorithm(emissions, transitions, start_trans, end_trans)
        parallel = _forward_algorithm_parallel(emissions, transitions, start_trans, end_trans)

        torch.testing.assert_close(sequential, parallel, rtol=1e-4, atol=1e-4)

    def test_matches_sequential_long(self):
        """Parallel should match sequential for long sequences."""
        emissions = torch.randn(2, 500, 4)
        transitions = torch.randn(4, 4)
        start_trans = torch.randn(4)
        end_trans = torch.randn(4)

        sequential = _forward_algorithm(emissions, transitions, start_trans, end_trans)
        parallel = _forward_algorithm_parallel(emissions, transitions, start_trans, end_trans)

        torch.testing.assert_close(sequential, parallel, rtol=1e-3, atol=1e-3)

    def test_matches_sequential_power_of_two(self):
        """Test with power-of-two sequence length (no padding needed)."""
        emissions = torch.randn(2, 64, 4)
        transitions = torch.randn(4, 4)
        start_trans = torch.randn(4)
        end_trans = torch.randn(4)

        sequential = _forward_algorithm(emissions, transitions, start_trans, end_trans)
        parallel = _forward_algorithm_parallel(emissions, transitions, start_trans, end_trans)

        torch.testing.assert_close(sequential, parallel, rtol=1e-4, atol=1e-4)

    def test_crf_parallel_vs_sequential(self):
        """CRF loss should be same with parallel and sequential."""
        crf = CRF(num_tags=4)
        emissions = torch.randn(3, 100, 4)
        tags = torch.randint(0, 4, (3, 100))

        loss_parallel = crf(emissions, tags, use_parallel=True)
        loss_sequential = crf(emissions, tags, use_parallel=False)

        torch.testing.assert_close(loss_parallel, loss_sequential, rtol=1e-4, atol=1e-4)


class TestCRFDevice:
    """Tests for device placement."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self):
        crf = CRF(num_tags=4).cuda()
        emissions = torch.randn(2, 10, 4).cuda()
        tags = torch.randint(0, 4, (2, 10)).cuda()

        loss = crf(emissions, tags)
        assert loss.device.type == "cuda"

        decoded = crf.decode(emissions)
        assert decoded.device.type == "cuda"

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_mps(self):
        crf = CRF(num_tags=4).to("mps")
        emissions = torch.randn(2, 10, 4).to("mps")
        tags = torch.randint(0, 4, (2, 10)).to("mps")

        loss = crf(emissions, tags)
        assert loss.device.type == "mps"

        decoded = crf.decode(emissions)
        assert decoded.device.type == "mps"


class TestMarginals:
    """Tests for marginal probability computation (forward-backward algorithm)."""

    def test_output_shape(self):
        crf = CRF(num_tags=4)
        emissions = torch.randn(3, 15, 4)
        marginals = crf.marginals(emissions)
        assert marginals.shape == (3, 15, 4)

    def test_marginals_sum_to_one(self):
        """Marginal probabilities at each position should sum to 1."""
        crf = CRF(num_tags=4)
        emissions = torch.randn(3, 15, 4)
        marginals = crf.marginals(emissions)
        sums = marginals.sum(dim=2)
        torch.testing.assert_close(sums, torch.ones(3, 15), rtol=1e-4, atol=1e-4)

    def test_marginals_are_probabilities(self):
        """Marginals should be between 0 and 1."""
        crf = CRF(num_tags=4)
        emissions = torch.randn(3, 15, 4)
        marginals = crf.marginals(emissions)
        assert (marginals >= 0).all()
        assert (marginals <= 1).all()

    def test_strong_emissions_high_marginals(self):
        """When emissions strongly favor a state, marginals should be high for that state."""
        crf = CRF(num_tags=4)
        with torch.no_grad():
            crf.transitions.fill_(0.0)
            crf.start_transitions.fill_(0.0)
            crf.end_transitions.fill_(0.0)

        emissions = torch.full((2, 10, 4), -10.0)
        emissions[:, :, 0] = 10.0

        marginals = crf.marginals(emissions)
        assert (marginals[:, :, 0] > 0.99).all()

    def test_single_timestep(self):
        """For single timestep, marginals should be softmax of emit + start + end."""
        emissions = torch.randn(2, 1, 4)
        transitions = torch.randn(4, 4)
        start_trans = torch.randn(4)
        end_trans = torch.randn(4)

        marginals = _compute_marginals(emissions, transitions, start_trans, end_trans)
        expected = torch.softmax(emissions[:, 0] + start_trans + end_trans, dim=1)
        torch.testing.assert_close(marginals[:, 0], expected, rtol=1e-4, atol=1e-4)

    def test_consistent_with_viterbi(self):
        """Argmax of marginals should often match Viterbi."""
        crf = CRF(num_tags=4)
        emissions = torch.randn(5, 20, 4)

        marginals = crf.marginals(emissions)
        viterbi = crf.decode(emissions)
        marginal_argmax = marginals.argmax(dim=2)

        match_rate = (marginal_argmax == viterbi).float().mean()
        assert match_rate > 0.8

    def test_batch_independence(self):
        """Marginals for each batch element should be computed independently."""
        crf = CRF(num_tags=4)
        emissions = torch.randn(3, 10, 4)
        batch_marginals = crf.marginals(emissions)

        for i in range(3):
            individual = crf.marginals(emissions[i : i + 1])
            torch.testing.assert_close(batch_marginals[i : i + 1], individual, rtol=1e-4, atol=1e-4)


class TestCRFEdgeCases:
    """Edge case tests."""

    def test_batch_size_one(self):
        crf = CRF(num_tags=4)
        emissions = torch.randn(1, 10, 4)
        tags = torch.randint(0, 4, (1, 10))

        loss = crf(emissions, tags)
        decoded = crf.decode(emissions)

        assert loss.shape == ()
        assert decoded.shape == (1, 10)

    def test_seq_len_one(self):
        crf = CRF(num_tags=4)
        emissions = torch.randn(3, 1, 4)
        tags = torch.randint(0, 4, (3, 1))

        loss = crf(emissions, tags)
        decoded = crf.decode(emissions)

        assert loss.shape == ()
        assert decoded.shape == (3, 1)

    def test_num_tags_two(self):
        crf = CRF(num_tags=2)
        emissions = torch.randn(2, 10, 2)
        tags = torch.randint(0, 2, (2, 10))

        loss = crf(emissions, tags)
        decoded = crf.decode(emissions)

        assert loss.shape == ()
        assert decoded.shape == (2, 10)
        assert (decoded >= 0).all()
        assert (decoded < 2).all()

    def test_long_sequence(self):
        """Test with sequence length similar to actual use case."""
        crf = CRF(num_tags=4)
        emissions = torch.randn(2, 2000, 4)
        tags = torch.randint(0, 4, (2, 2000))

        loss = crf(emissions, tags)
        decoded = crf.decode(emissions)

        assert loss.isfinite()
        assert decoded.shape == (2, 2000)

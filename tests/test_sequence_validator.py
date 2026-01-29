"""
Unit tests for cardiac cycle sequence validation.
"""

import numpy as np
import torch

from hss.utils.sequence_validator import (
    CardiacCycleValidator,
    validate_and_correct_predictions,
)


class TestCardiacCycleValidator:
    """Test suite for CardiacCycleValidator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = CardiacCycleValidator()

    def test_transition_matrix_shape(self):
        """Test that transition matrix has correct shape."""
        assert self.validator.transition_matrix.shape == (4, 4)

    def test_valid_transitions(self):
        """Test that valid transitions are recognized."""
        # Valid forward transitions: 1->2, 2->3, 3->4, 4->1
        assert self.validator.is_valid_transition(1, 2)
        assert self.validator.is_valid_transition(2, 3)
        assert self.validator.is_valid_transition(3, 4)
        assert self.validator.is_valid_transition(4, 1)

    def test_invalid_transitions(self):
        """Test that invalid transitions are rejected."""
        # Invalid transitions
        assert not self.validator.is_valid_transition(1, 3)  # Skip state
        assert not self.validator.is_valid_transition(1, 4)
        assert not self.validator.is_valid_transition(2, 1)  # Backward
        assert not self.validator.is_valid_transition(2, 4)
        assert not self.validator.is_valid_transition(3, 1)
        assert not self.validator.is_valid_transition(3, 2)
        assert not self.validator.is_valid_transition(4, 2)
        assert not self.validator.is_valid_transition(4, 3)

    def test_self_transitions(self):
        """Test that self-transitions are valid (staying in same state)."""
        assert self.validator.is_valid_transition(1, 1)
        assert self.validator.is_valid_transition(2, 2)
        assert self.validator.is_valid_transition(3, 3)
        assert self.validator.is_valid_transition(4, 4)

    def test_out_of_bounds_transitions(self):
        """Test that out-of-bounds labels are rejected."""
        assert not self.validator.is_valid_transition(0, 1)
        assert not self.validator.is_valid_transition(1, 5)
        assert not self.validator.is_valid_transition(-1, 2)

    def test_validate_valid_sequence(self):
        """Test validation of a valid sequence."""
        valid_sequence = np.array([1, 2, 3, 4, 1, 2, 3, 4])
        is_valid, invalid_positions = self.validator.validate_sequence(valid_sequence)

        assert is_valid
        assert len(invalid_positions) == 0

    def test_validate_valid_sequence_with_self_transitions(self):
        """Test validation of a sequence with self-transitions."""
        # Sequence with self-transitions (staying in same state)
        valid_sequence = np.array([1, 1, 2, 2, 2, 3, 3, 4, 4, 1])
        is_valid, invalid_positions = self.validator.validate_sequence(valid_sequence)

        assert is_valid
        assert len(invalid_positions) == 0

    def test_validate_invalid_sequence(self):
        """Test validation of an invalid sequence."""
        # Sequence with invalid transitions: 1->3 (skip), 3->4 (valid), 4->2 (backward), 2->1 (backward)
        invalid_sequence = np.array([1, 3, 4, 2, 1])
        is_valid, invalid_positions = self.validator.validate_sequence(invalid_sequence)

        assert not is_valid
        assert len(invalid_positions) == 3
        assert 0 in invalid_positions  # 1->3 at position 0
        assert 2 in invalid_positions  # 4->2 at position 2
        assert 3 in invalid_positions  # 2->1 at position 3

    def test_correct_sequence_viterbi(self):
        """Test Viterbi-based sequence correction."""
        # Create log probabilities that prefer an invalid sequence
        seq_len = 8
        log_probs = np.zeros((seq_len, 4))

        # Set high probabilities for the valid sequence 1->2->3->4->1->2->3->4
        for i in range(seq_len):
            label = i % 4
            log_probs[i, label] = 0.0  # High log-prob
            # Set lower probs for other states
            for j in range(4):
                if j != label:
                    log_probs[i, j] = -10.0

        corrected = self.validator.correct_sequence_viterbi(log_probs)

        # Check that corrected sequence is valid
        is_valid, _ = self.validator.validate_sequence(corrected)
        assert is_valid

        # Check expected sequence
        expected = np.array([1, 2, 3, 4, 1, 2, 3, 4])
        np.testing.assert_array_equal(corrected, expected)

    def test_viterbi_with_conflicting_probabilities(self):
        """Test Viterbi when probabilities suggest invalid transitions."""
        # Create log probabilities that would prefer invalid sequence
        # but Viterbi should find the best valid sequence
        log_probs = np.array(
            [
                [0.0, -5.0, -5.0, -5.0],  # Strongly prefers 1
                [-5.0, -5.0, 0.0, -5.0],  # Strongly prefers 3 (invalid from 1)
                [-5.0, -5.0, -5.0, 0.0],  # Strongly prefers 4
                [0.0, -5.0, -5.0, -5.0],  # Strongly prefers 1
            ]
        )

        corrected = self.validator.correct_sequence_viterbi(log_probs)

        # Should find a valid path, even if not the highest probability
        is_valid, _ = self.validator.validate_sequence(corrected)
        assert is_valid

        # Viterbi finds optimal path 2->3->4->1 (score: -5+0+0+0 = -5)
        # which is better than 1->2->3->4 (score: 0-5+0-5 = -10)
        expected = np.array([2, 3, 4, 1])
        np.testing.assert_array_equal(corrected, expected)

    def test_viterbi_return_score(self):
        """Test that Viterbi can return the score."""
        log_probs = np.array(
            [
                [0.0, -1.0, -2.0, -3.0],
                [-3.0, 0.0, -1.0, -2.0],
                [-2.0, -3.0, 0.0, -1.0],
                [-1.0, -2.0, -3.0, 0.0],
            ]
        )

        corrected, score = self.validator.correct_sequence_viterbi(log_probs, return_score=True)

        assert isinstance(corrected, np.ndarray)
        assert isinstance(score, (float, np.floating))
        assert corrected.shape == (4,)

        # Check that sequence is valid
        is_valid, _ = self.validator.validate_sequence(corrected)
        assert is_valid


class TestValidateAndCorrectPredictions:
    """Test suite for the high-level validation function."""

    def test_with_numpy_single_sequence(self):
        """Test with numpy array, single sequence."""
        log_probs = np.array(
            [
                [0.0, -1.0, -2.0, -3.0],
                [-3.0, 0.0, -1.0, -2.0],
                [-2.0, -3.0, 0.0, -1.0],
                [-1.0, -2.0, -3.0, 0.0],
            ]
        )

        corrected = validate_and_correct_predictions(log_probs)

        assert isinstance(corrected, np.ndarray)
        assert corrected.shape == (4,)

        validator = CardiacCycleValidator()
        is_valid, _ = validator.validate_sequence(corrected)
        assert is_valid

    def test_with_torch_single_sequence(self):
        """Test with torch tensor, single sequence."""
        log_probs = torch.tensor(
            [
                [0.0, -1.0, -2.0, -3.0],
                [-3.0, 0.0, -1.0, -2.0],
                [-2.0, -3.0, 0.0, -1.0],
                [-1.0, -2.0, -3.0, 0.0],
            ]
        )

        corrected = validate_and_correct_predictions(log_probs)

        assert isinstance(corrected, torch.Tensor)
        assert corrected.shape == (4,)

        validator = CardiacCycleValidator()
        is_valid, _ = validator.validate_sequence(corrected.numpy())
        assert is_valid

    def test_with_batch(self):
        """Test with batched input."""
        batch_size = 3
        seq_len = 4
        log_probs = np.zeros((batch_size, seq_len, 4))

        # Different sequences for each batch
        for b in range(batch_size):
            for t in range(seq_len):
                label = (t + b) % 4
                log_probs[b, t, label] = 0.0
                for j in range(4):
                    if j != label:
                        log_probs[b, t, j] = -5.0

        corrected = validate_and_correct_predictions(log_probs)

        assert corrected.shape == (batch_size, seq_len)

        validator = CardiacCycleValidator()
        for b in range(batch_size):
            is_valid, _ = validator.validate_sequence(corrected[b])
            assert is_valid


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_sequence(self):
        """Test handling of empty sequence."""
        validator = CardiacCycleValidator()
        empty = np.array([])

        is_valid, invalid_positions = validator.validate_sequence(empty)
        assert is_valid
        assert len(invalid_positions) == 0

    def test_single_element_sequence(self):
        """Test handling of single element sequence."""
        validator = CardiacCycleValidator()
        single = np.array([1])

        is_valid, invalid_positions = validator.validate_sequence(single)
        assert is_valid
        assert len(invalid_positions) == 0

    def test_long_valid_sequence(self):
        """Test with a long valid sequence (multiple cycles)."""
        validator = CardiacCycleValidator()

        # Create 3 complete cardiac cycles
        long_sequence = np.tile([1, 2, 3, 4], 3)

        is_valid, invalid_positions = validator.validate_sequence(long_sequence)
        assert is_valid
        assert len(invalid_positions) == 0

    def test_all_same_probabilities(self):
        """Test Viterbi when all probabilities are equal."""
        validator = CardiacCycleValidator()

        # All log probs are the same
        log_probs = np.zeros((8, 4))

        corrected = validator.correct_sequence_viterbi(log_probs)

        # Should still produce a valid sequence
        is_valid, _ = validator.validate_sequence(corrected)
        assert is_valid

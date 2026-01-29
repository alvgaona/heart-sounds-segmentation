"""
Label sequence validator for heart sound segmentation.

Enforces the cardiac cycle constraint: S1 -> Systole -> S2 -> Diastole -> S1
Labels: 1 -> 2 -> 3 -> 4 -> 1
"""

from typing import Literal, overload

import numpy as np
import torch


class CardiacCycleValidator:
    """
    Validates and corrects label sequences to follow the cardiac cycle pattern.

    The cardiac cycle must follow: 1 -> 2 -> 3 -> 4 -> 1
    Where:
        1: S1 (first heart sound)
        2: Systole (interval between S1 and S2)
        3: S2 (second heart sound)
        4: Diastole (interval between S2 and S1)
    """

    def __init__(self):
        """Initialize the validator with transition constraints."""
        # Define valid transitions (1-indexed to match the labels)
        # transition_matrix[i][j] = True if transition from i to j is valid
        self.transition_matrix = self._build_transition_matrix()

    def _build_transition_matrix(self) -> np.ndarray:
        """
        Build a transition matrix for valid cardiac cycle transitions.

        Returns:
            np.ndarray: 4x4 boolean matrix where [i][j] indicates if
                       transition from label i+1 to label j+1 is valid
        """
        # Initialize all transitions as invalid
        matrix = np.zeros((4, 4), dtype=bool)

        # Self-transitions (staying in the same state)
        matrix[0, 0] = True  # S1 -> S1
        matrix[1, 1] = True  # Systole -> Systole
        matrix[2, 2] = True  # S2 -> S2
        matrix[3, 3] = True  # Diastole -> Diastole

        # Forward transitions (0-indexed for array access):
        matrix[0, 1] = True  # S1 -> Systole
        matrix[1, 2] = True  # Systole -> S2
        matrix[2, 3] = True  # S2 -> Diastole
        matrix[3, 0] = True  # Diastole -> S1

        return matrix

    def is_valid_transition(self, from_label: int, to_label: int) -> bool:
        """
        Check if a transition between two labels is valid.

        Args:
            from_label: Starting label (1-4)
            to_label: Ending label (1-4)

        Returns:
            bool: True if transition is valid, False otherwise
        """
        # Convert to 0-indexed
        from_idx = from_label - 1
        to_idx = to_label - 1

        # Check bounds
        if not (0 <= from_idx < 4 and 0 <= to_idx < 4):
            return False

        return self.transition_matrix[from_idx, to_idx]

    def validate_sequence(self, labels: np.ndarray) -> tuple[bool, list[int]]:
        """
        Validate a sequence of labels.

        Args:
            labels: Array of labels (1-4)

        Returns:
            tuple: (is_valid, invalid_positions)
                - is_valid: True if sequence is valid
                - invalid_positions: List of indices where invalid transitions occur
        """
        invalid_positions = []

        for i in range(len(labels) - 1):
            if not self.is_valid_transition(labels[i], labels[i + 1]):
                invalid_positions.append(i)

        return len(invalid_positions) == 0, invalid_positions

    @overload
    def correct_sequence_viterbi(self, log_probs: np.ndarray, return_score: Literal[False] = ...) -> np.ndarray: ...

    @overload
    def correct_sequence_viterbi(
        self, log_probs: np.ndarray, return_score: Literal[True]
    ) -> tuple[np.ndarray, float]: ...

    def correct_sequence_viterbi(
        self, log_probs: np.ndarray, return_score: bool = False
    ) -> np.ndarray | tuple[np.ndarray, float]:
        """
        Correct a sequence using Viterbi algorithm with transition constraints.

        This finds the most likely valid sequence given the model's output
        probabilities and the cardiac cycle constraints.

        Args:
            log_probs: Log probabilities from model (sequence_length, 4)
                      Each row contains log probabilities for classes 0-3
                      (corresponding to labels 1-4)
            return_score: If True, return the total score along with predictions

        Returns:
            If return_score is False:
                np.ndarray: Corrected label sequence (1-4)
            If return_score is True:
                tuple: (corrected_labels, total_score)
        """
        seq_len, num_classes = log_probs.shape
        assert num_classes == 4, "Expected 4 classes for cardiac cycle"

        # Viterbi algorithm (vectorized)
        # viterbi[t][s] = max log probability of reaching state s at time t
        viterbi = np.full((seq_len, num_classes), -np.inf)
        # backpointer[t][s] = previous state that led to state s at time t
        backpointer = np.zeros((seq_len, num_classes), dtype=int)

        # Initialize: first timestep can be any state
        viterbi[0] = log_probs[0]

        # Precompute transition mask (invalid transitions become -inf)
        trans_mask = np.where(self.transition_matrix, 0.0, -np.inf)

        # Forward pass (vectorized over states)
        for t in range(1, seq_len):
            # scores[i, j] = viterbi[t-1, i] + trans_mask[i, j] + log_probs[t, j]
            # Shape: (num_classes, num_classes)
            scores = viterbi[t - 1, :, np.newaxis] + trans_mask + log_probs[t, np.newaxis, :]

            # Best previous state for each current state
            backpointer[t] = np.argmax(scores, axis=0)
            viterbi[t] = np.max(scores, axis=0)

        # Backtrack to find the best path
        best_path = np.zeros(seq_len, dtype=int)
        best_path[-1] = np.argmax(viterbi[-1])
        total_score = viterbi[-1, best_path[-1]]

        for t in range(seq_len - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]

        # Convert from 0-indexed to 1-indexed labels
        corrected_labels = best_path + 1

        if return_score:
            return corrected_labels, total_score
        return corrected_labels

    def correct_batch_viterbi(self, log_probs: np.ndarray) -> np.ndarray:
        """
        Correct a batch of sequences using vectorized Viterbi algorithm.

        Args:
            log_probs: Log probabilities from model (batch_size, sequence_length, 4)

        Returns:
            np.ndarray: Corrected label sequences (batch_size, sequence_length) with labels 1-4
        """
        batch_size, seq_len, num_classes = log_probs.shape
        assert num_classes == 4, "Expected 4 classes for cardiac cycle"

        # Viterbi algorithm (vectorized over batch and states)
        # viterbi[b, t, s] = max log probability of reaching state s at time t for batch b
        viterbi = np.full((batch_size, seq_len, num_classes), -np.inf)
        backpointer = np.zeros((batch_size, seq_len, num_classes), dtype=int)

        # Initialize: first timestep can be any state
        viterbi[:, 0, :] = log_probs[:, 0, :]

        # Precompute transition mask
        trans_mask = np.where(self.transition_matrix, 0.0, -np.inf)

        # Forward pass (vectorized over batch and states)
        for t in range(1, seq_len):
            # scores[b, i, j] = viterbi[b, t-1, i] + trans_mask[i, j] + log_probs[b, t, j]
            # Shape: (batch_size, num_classes, num_classes)
            scores = viterbi[:, t - 1, :, np.newaxis] + trans_mask[np.newaxis, :, :] + log_probs[:, t, np.newaxis, :]

            backpointer[:, t, :] = np.argmax(scores, axis=1)
            viterbi[:, t, :] = np.max(scores, axis=1)

        # Backtrack to find best paths
        best_paths = np.zeros((batch_size, seq_len), dtype=int)
        best_paths[:, -1] = np.argmax(viterbi[:, -1, :], axis=1)

        for t in range(seq_len - 2, -1, -1):
            for b in range(batch_size):
                best_paths[b, t] = backpointer[b, t + 1, best_paths[b, t + 1]]

        # Convert from 0-indexed to 1-indexed labels
        return best_paths + 1

    def correct_sequence_greedy(self, labels: np.ndarray) -> np.ndarray:
        """
        Correct a sequence using a greedy forward pass.

        This is simpler than Viterbi but doesn't use probability information.
        It just fixes invalid transitions by adjusting the next label.

        Args:
            labels: Array of labels (1-4)

        Returns:
            np.ndarray: Corrected label sequence (1-4)
        """
        if len(labels) == 0:
            return labels

        corrected = labels.copy()

        for i in range(len(corrected) - 1):
            current_label = corrected[i]
            next_label = corrected[i + 1]

            if not self.is_valid_transition(current_label, next_label):
                # Fix by setting next label to the only valid transition
                # From label k, the only valid next label is (k % 4) + 1
                corrected[i + 1] = (current_label % 4) + 1

        return corrected


def validate_and_correct_predictions(
    log_probs: torch.Tensor | np.ndarray, method: str = "viterbi"
) -> torch.Tensor | np.ndarray:
    """
    Validate and correct predictions to follow cardiac cycle constraints.

    Args:
        log_probs: Log probabilities from model
                  Shape: (batch_size, sequence_length, 4) or (sequence_length, 4)
        method: Correction method - "viterbi" or "greedy"

    Returns:
        Corrected predictions (1-4). Returns torch.Tensor if input was tensor,
        np.ndarray otherwise. Shape: (batch_size, sequence_length) or (sequence_length,)
    """
    validator = CardiacCycleValidator()

    # Convert to numpy if needed
    if isinstance(log_probs, torch.Tensor):
        log_probs_np = log_probs.detach().cpu().numpy()
        return_torch = True
    else:
        log_probs_np = log_probs
        return_torch = False

    # Handle batch dimension
    if log_probs_np.ndim == 3:
        if method == "viterbi":
            result = validator.correct_batch_viterbi(log_probs_np)
        elif method == "greedy":
            batch_size, seq_len, num_classes = log_probs_np.shape
            corrected_batch = np.zeros((batch_size, seq_len), dtype=int)
            for i in range(batch_size):
                uncorrected = np.argmax(log_probs_np[i], axis=1) + 1
                corrected_batch[i] = validator.correct_sequence_greedy(uncorrected)
            result = corrected_batch
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        # Single sequence
        if method == "viterbi":
            result = validator.correct_sequence_viterbi(log_probs_np)
        elif method == "greedy":
            uncorrected = np.argmax(log_probs_np, axis=1) + 1
            result = validator.correct_sequence_greedy(uncorrected)
        else:
            raise ValueError(f"Unknown method: {method}")

    if return_torch:
        return torch.from_numpy(result)
    return result

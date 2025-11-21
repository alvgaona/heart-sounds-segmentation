"""
Example demonstrating cardiac cycle sequence validation.

This script shows how the sequence validator enforces the cardiac cycle constraint:
Labels must follow the pattern: 1 -> 2 -> 3 -> 4 -> 1 (repeating)

Where:
    1: S1 (first heart sound)
    2: Systole (interval between S1 and S2)
    3: S2 (second heart sound)
    4: Diastole (interval between S2 and S1)
"""

import numpy as np

from hss.utils.sequence_validator import CardiacCycleValidator, validate_and_correct_predictions


def example_valid_transitions():
    """Demonstrate valid and invalid transitions."""
    print("=" * 60)
    print("EXAMPLE 1: Valid and Invalid Transitions")
    print("=" * 60)

    validator = CardiacCycleValidator()

    print("\nValid transitions (cardiac cycle order):")
    valid_transitions = [(1, 2), (2, 3), (3, 4), (4, 1)]
    for from_label, to_label in valid_transitions:
        is_valid = validator.is_valid_transition(from_label, to_label)
        print(f"  {from_label} -> {to_label}: {'✓ Valid' if is_valid else '✗ Invalid'}")

    print("\nInvalid transitions (violate cardiac cycle):")
    invalid_transitions = [(1, 3), (4, 2), (2, 1), (3, 1), (1, 1)]
    for from_label, to_label in invalid_transitions:
        is_valid = validator.is_valid_transition(from_label, to_label)
        print(f"  {from_label} -> {to_label}: {'✓ Valid' if is_valid else '✗ Invalid'}")


def example_sequence_validation():
    """Demonstrate sequence validation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Sequence Validation")
    print("=" * 60)

    validator = CardiacCycleValidator()

    # Valid sequence
    valid_sequence = np.array([1, 2, 3, 4, 1, 2, 3, 4])
    is_valid, invalid_positions = validator.validate_sequence(valid_sequence)
    print(f"\nSequence: {valid_sequence}")
    print(f"Valid: {is_valid}")

    # Invalid sequence
    invalid_sequence = np.array([1, 3, 4, 2, 1])
    is_valid, invalid_positions = validator.validate_sequence(invalid_sequence)
    print(f"\nSequence: {invalid_sequence}")
    print(f"Valid: {is_valid}")
    print(f"Invalid transitions at positions: {invalid_positions}")
    for pos in invalid_positions:
        from_label = invalid_sequence[pos]
        to_label = invalid_sequence[pos + 1]
        print(f"  Position {pos}: {from_label} -> {to_label}")


def example_greedy_correction():
    """Demonstrate greedy sequence correction."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Greedy Sequence Correction")
    print("=" * 60)

    validator = CardiacCycleValidator()

    # Start with an invalid sequence
    invalid_sequence = np.array([1, 3, 4, 2, 1, 2, 3, 1])
    print(f"\nOriginal sequence: {invalid_sequence}")

    is_valid, invalid_positions = validator.validate_sequence(invalid_sequence)
    print(f"Valid: {is_valid}")
    print(f"Invalid transitions at positions: {invalid_positions}")

    # Correct the sequence
    corrected = validator.correct_sequence_greedy(invalid_sequence)
    print(f"\nCorrected sequence: {corrected}")

    is_valid, invalid_positions = validator.validate_sequence(corrected)
    print(f"Valid: {is_valid}")


def example_viterbi_correction():
    """Demonstrate Viterbi-based correction with probabilities."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Viterbi Correction (Probability-Based)")
    print("=" * 60)

    validator = CardiacCycleValidator()

    # Create log probabilities that would suggest an invalid sequence
    # but Viterbi will find the best valid path
    print("\nScenario: Model predicts high probabilities for invalid sequence")
    print("Viterbi algorithm finds the best VALID sequence given these probabilities")

    # Sequence length of 8 timesteps
    log_probs = np.array([
        [0.0, -5.0, -5.0, -5.0],   # t=0: Strongly prefers label 1 (S1)
        [-5.0, 0.0, -5.0, -5.0],   # t=1: Strongly prefers label 2 (Systole) ✓ valid from 1
        [-5.0, -5.0, 0.0, -5.0],   # t=2: Strongly prefers label 3 (S2) ✓ valid from 2
        [-5.0, -5.0, -5.0, 0.0],   # t=3: Strongly prefers label 4 (Diastole) ✓ valid from 3
        [0.0, -5.0, -5.0, -5.0],   # t=4: Strongly prefers label 1 (S1) ✓ valid from 4
        [-5.0, 0.0, -5.0, -5.0],   # t=5: Strongly prefers label 2 (Systole) ✓ valid from 1
        [-5.0, -5.0, 0.0, -5.0],   # t=6: Strongly prefers label 3 (S2) ✓ valid from 2
        [-5.0, -5.0, -5.0, 0.0],   # t=7: Strongly prefers label 4 (Diastole) ✓ valid from 3
    ])

    print("\nLog probabilities (each row = probabilities for labels 1-4):")
    for t, probs in enumerate(log_probs):
        max_idx = np.argmax(probs)
        print(f"  t={t}: max at label {max_idx + 1} (log_prob={probs[max_idx]:.1f})")

    # Get the unconstrained predictions (what model would predict without constraints)
    unconstrained_preds = np.argmax(log_probs, axis=1) + 1
    print(f"\nUnconstrained predictions: {unconstrained_preds}")

    is_valid, _ = validator.validate_sequence(unconstrained_preds)
    print(f"Are unconstrained predictions valid? {is_valid}")

    # Apply Viterbi correction
    corrected, score = validator.correct_sequence_viterbi(log_probs, return_score=True)
    print(f"\nViterbi corrected sequence: {corrected}")
    print(f"Total log probability score: {score:.2f}")

    is_valid, _ = validator.validate_sequence(corrected)
    print(f"Are corrected predictions valid? {is_valid}")


def example_conflicting_probabilities():
    """Demonstrate Viterbi with conflicting probabilities."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Viterbi with Conflicting Probabilities")
    print("=" * 60)

    validator = CardiacCycleValidator()

    # Create probabilities that would prefer an invalid sequence
    print("\nScenario: Model predictions strongly prefer INVALID transitions")
    print("Viterbi must find best valid path even if it has lower probability")

    log_probs = np.array([
        [0.0, -2.0, -5.0, -5.0],   # t=0: Prefers 1 (S1)
        [-5.0, -5.0, 0.0, -2.0],   # t=1: Prefers 3 (S2) but 3 invalid from 1! Must choose 2
        [-5.0, -5.0, -2.0, 0.0],   # t=2: Prefers 4 (Diastole) but 4 invalid from 2! Must choose 3
        [0.0, -2.0, -5.0, -5.0],   # t=3: Prefers 1 (S1) but 1 invalid from 3! Must choose 4
    ])

    print("\nLog probabilities:")
    for t, probs in enumerate(log_probs):
        max_idx = np.argmax(probs)
        second_max_idx = np.argsort(probs)[-2]
        print(
            f"  t={t}: prefers label {max_idx + 1} ({probs[max_idx]:.1f}), "
            f"2nd choice: label {second_max_idx + 1} ({probs[second_max_idx]:.1f})"
        )

    unconstrained_preds = np.argmax(log_probs, axis=1) + 1
    print(f"\nUnconstrained (greedy) predictions: {unconstrained_preds}")

    is_valid, invalid_pos = validator.validate_sequence(unconstrained_preds)
    print(f"Valid? {is_valid}")
    if not is_valid:
        print(f"Invalid transitions at positions: {invalid_pos}")

    # Apply Viterbi
    corrected = validator.correct_sequence_viterbi(log_probs)
    print(f"\nViterbi corrected sequence: {corrected}")

    is_valid, _ = validator.validate_sequence(corrected)
    print(f"Valid? {is_valid}")
    print("\nNote: Viterbi chose valid transitions even though they had lower probabilities!")


def example_batch_processing():
    """Demonstrate batch processing."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Batch Processing")
    print("=" * 60)

    # Simulate batch of predictions
    batch_size = 3
    seq_len = 4

    # Create random log probabilities for a batch
    np.random.seed(42)
    log_probs = np.random.randn(batch_size, seq_len, 4)

    print(f"\nProcessing batch of {batch_size} sequences, each with {seq_len} timesteps")

    # Apply validation to the entire batch
    corrected = validate_and_correct_predictions(log_probs, method="viterbi")

    print("\nResults:")
    validator = CardiacCycleValidator()
    for i in range(batch_size):
        unconstrained = np.argmax(log_probs[i], axis=1) + 1
        is_valid_before, _ = validator.validate_sequence(unconstrained)
        is_valid_after, _ = validator.validate_sequence(corrected[i])

        print(f"  Sequence {i + 1}:")
        print(f"    Unconstrained: {unconstrained} (valid: {is_valid_before})")
        print(f"    Constrained:   {corrected[i]} (valid: {is_valid_after})")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("CARDIAC CYCLE SEQUENCE VALIDATION EXAMPLES")
    print("=" * 60)
    print("\nConstraint: Labels must follow 1 -> 2 -> 3 -> 4 -> 1 (cardiac cycle)")
    print("  1: S1 (first heart sound)")
    print("  2: Systole")
    print("  3: S2 (second heart sound)")
    print("  4: Diastole")

    example_valid_transitions()
    example_sequence_validation()
    example_greedy_correction()
    example_viterbi_correction()
    example_conflicting_probabilities()
    example_batch_processing()

    print("\n" + "=" * 60)
    print("EXAMPLES COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()

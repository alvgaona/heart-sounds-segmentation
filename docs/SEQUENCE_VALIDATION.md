# Cardiac Cycle Sequence Validation

## Overview

This feature enforces the natural cardiac cycle ordering constraint on heart sound segmentation predictions. The cardiac cycle follows a strict physiological sequence that must be respected:

```
S1 → Systole → S2 → Diastole → S1 (repeat)
 1 →    2    →  3 →     4     →  1
```

## The Problem

Without constraints, neural network models can predict physiologically impossible label sequences, such as:
- **1 → 3** (S1 directly to S2, skipping Systole)
- **4 → 2** (Diastole directly to Systole, skipping S1)
- **2 → 1** (Going backwards in the cycle)

These invalid transitions violate the fundamental structure of the cardiac cycle.

## The Solution

The sequence validator provides two methods to enforce valid transitions:

### 1. Viterbi Algorithm (Recommended)
Uses dynamic programming to find the most likely **valid** sequence given the model's output probabilities. This method:
- Considers all possible valid paths through the sequence
- Maximizes the total probability while respecting transition constraints
- Always produces a valid cardiac cycle sequence

### 2. Greedy Correction
A simpler approach that:
- Takes the model's predictions
- Corrects invalid transitions by forcing the next valid state
- Faster but may not find the globally optimal solution

## Valid Transitions

| From State | To State | Valid? | Physiological Meaning |
|------------|----------|--------|----------------------|
| 1 (S1) | 2 (Systole) | ✓ | Normal progression |
| 2 (Systole) | 3 (S2) | ✓ | Normal progression |
| 3 (S2) | 4 (Diastole) | ✓ | Normal progression |
| 4 (Diastole) | 1 (S1) | ✓ | Cycle restart |
| Any other transition | | ✗ | Physiologically impossible |

## Usage

### In Training Pipeline

The validation is automatically integrated into the training pipeline in `main.py`:

```python
from hss.utils.sequence_validator import validate_and_correct_predictions

# During model initialization
model = LitModel(
    input_size=44,
    batch_size=batch_size,
    device=device,
    use_sequence_constraints=True  # Enable validation (default: True)
)
```

When enabled, the model will:
1. Train normally (no constraints on gradients)
2. During validation/testing, compute both:
   - Standard metrics (unconstrained predictions)
   - Constrained metrics (sequence-validated predictions)

### Standalone Usage

You can also use the validator independently:

```python
import numpy as np
from hss.utils.sequence_validator import CardiacCycleValidator, validate_and_correct_predictions

# Method 1: Using the high-level function
log_probs = model_output  # Shape: (batch_size, seq_len, 4)
corrected_labels = validate_and_correct_predictions(log_probs, method="viterbi")

# Method 2: Using the validator class directly
validator = CardiacCycleValidator()

# Check if a transition is valid
is_valid = validator.is_valid_transition(from_label=1, to_label=2)  # True
is_valid = validator.is_valid_transition(from_label=1, to_label=3)  # False

# Validate a sequence
labels = np.array([1, 2, 3, 4, 1, 2, 3, 4])
is_valid, invalid_positions = validator.validate_sequence(labels)

# Correct a sequence using Viterbi
log_probs = np.array([...])  # Shape: (seq_len, 4)
corrected = validator.correct_sequence_viterbi(log_probs)

# Correct using greedy method
invalid_labels = np.array([1, 3, 4, 2, 1])
corrected = validator.correct_sequence_greedy(invalid_labels)
```

### Example Scripts

Run the example script to see the validator in action:

```bash
python examples/sequence_validation_example.py
```

This demonstrates:
1. Valid vs invalid transitions
2. Sequence validation
3. Greedy correction
4. Viterbi correction with probabilities
5. Handling conflicting probabilities
6. Batch processing

## How It Works

### Viterbi Algorithm

The Viterbi algorithm is a dynamic programming approach that finds the most probable path through a state sequence while respecting transition constraints.

**Algorithm Steps:**
1. **Initialization**: Start with initial state probabilities from the model
2. **Forward Pass**: For each timestep, compute the maximum probability of reaching each state from valid previous states
3. **Backtracking**: Trace back from the final state to find the optimal path

**Mathematical Formulation:**
```
viterbi[t][s] = max over prev_states of:
    viterbi[t-1][prev_state] + log_prob[t][s]
    where transition(prev_state → s) is valid
```

### Greedy Correction

The greedy approach is simpler:
1. Start with the first predicted label
2. For each subsequent prediction:
   - If the transition is valid, keep it
   - If invalid, replace with the only valid next state
3. Continue until the end of the sequence

## Integration Details

### Model Architecture

The sequence validator integrates with the `LitModel` PyTorch Lightning module:

- **Training**: No constraints applied (model learns naturally)
- **Validation**: Computes both constrained and unconstrained metrics
- **Testing**: Computes both constrained and unconstrained metrics

### Metrics

When `use_sequence_constraints=True`, the model logs additional metrics:

**Standard Metrics:**
- `test_accuracy`, `test_precision`, `test_f1`, etc.

**Constrained Metrics:**
- `test_constrained_accuracy`
- `test_constrained_precision`
- `test_constrained_f1`
- etc.

This allows you to compare:
- Raw model performance (what the model predicts)
- Constrained performance (physiologically valid predictions)

## Performance Considerations

### Computational Complexity

- **Viterbi Algorithm**: O(seq_len × num_states²) = O(seq_len × 16)
  - With only 4 states and sparse transitions, this is very efficient
- **Greedy Correction**: O(seq_len)
  - Even faster but may produce suboptimal results

### Memory Usage

- Minimal overhead: stores transition matrix (4×4 boolean array)
- Viterbi requires temporary arrays of size (seq_len × 4)

### Runtime

On typical sequences (1000-2000 timesteps):
- **Viterbi**: ~1-2ms per sequence
- **Greedy**: ~0.1-0.5ms per sequence

Both are negligible compared to model inference time.

## Implementation Files

- **Core Module**: `hss/utils/sequence_validator.py`
  - `CardiacCycleValidator` class
  - `validate_and_correct_predictions()` function

- **Integration**: `main.py`
  - Modified `LitModel` class with constraint support

- **Tests**: `tests/test_sequence_validator.py`
  - Comprehensive unit tests

- **Examples**: `examples/sequence_validation_example.py`
  - Demonstration scripts

## Testing

Run the test suite:

```bash
pytest tests/test_sequence_validator.py -v
```

Test coverage includes:
- Valid/invalid transition detection
- Sequence validation
- Viterbi correction with various probability distributions
- Greedy correction
- Batch processing
- Edge cases (empty sequences, single elements, etc.)

## Future Enhancements

Potential improvements:
1. **CRF Layer**: Integrate a Conditional Random Field layer directly into the model
2. **Training with Constraints**: Add transition penalties to the loss function
3. **Configurable Transitions**: Support different cardiac cycle patterns (e.g., abnormal rhythms)
4. **Smoothing**: Add temporal smoothing to reduce label flickering

## References

1. Springer, D. et al. (2016). "Logistic Regression-HSMM-based Heart Sound Segmentation"
2. Viterbi, A. (1967). "Error bounds for convolutional codes and an asymptotically optimum decoding algorithm"
3. Rabiner, L. (1989). "A tutorial on hidden Markov models and selected applications in speech recognition"

## Citation

If you use this sequence validation feature in your research, please cite:

```bibtex
@software{heart_sounds_segmentation,
  author = {Gaona, Alvaro J.},
  title = {Heart Sounds Segmentation with Sequence Validation},
  year = {2025},
  url = {https://github.com/alvgaona/heart-sounds-segmentation}
}
```

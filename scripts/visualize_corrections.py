#!/usr/bin/env python3
"""Visualize uncorrected vs Viterbi-corrected predictions."""

import matplotlib.pyplot as plt
import scipy
import torch
from torchvision import transforms

from hss.datasets.heart_sounds import DavidSpringerHSS
from hss.model.lit_model import LitModel
from hss.transforms import FSST
from hss.utils.sequence_validator import CardiacCycleValidator, validate_and_correct_predictions


def main():
    sample_idx = 1  # Sample with 7 invalid transitions

    # Load model
    checkpoint_path = "lightning_logs/version_12/checkpoints/epoch=14-step=1920.ckpt"
    model = LitModel.load_from_checkpoint(
        checkpoint_path,
        input_size=44,
        batch_size=1,
        device=torch.device("cpu"),
        use_sequence_constraints=True,
        weights_only=False,
    )
    model.eval()
    model.to("cpu")

    # Load dataset
    transform = transforms.Compose(
        [
            FSST(
                1000,
                window=scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=False),
                truncate_freq=(25, 200),
                stack=True,
            ),
        ]
    )

    dataset = DavidSpringerHSS(
        "resources/data",
        download=False,
        framing=True,
        in_memory=True,
        transform=transform,
        verbose=False,
    )

    # Get sample
    x, y = dataset[sample_idx]
    x = x.unsqueeze(0)

    with torch.no_grad():
        logits = model(x).permute((0, 2, 1))
        uncorrected = torch.argmax(logits, dim=1).squeeze().numpy() + 1  # 1-indexed

        # Get corrected predictions
        log_probs = logits.permute((0, 2, 1))  # (batch, seq, classes)
        corrected = validate_and_correct_predictions(log_probs, method="viterbi").squeeze().numpy()

    ground_truth = y.numpy() + 1  # Convert to 1-indexed

    # Find invalid positions
    validator = CardiacCycleValidator()
    _, invalid_pos = validator.validate_sequence(uncorrected)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    time = range(len(ground_truth))
    labels = ["S1", "Systole", "S2", "Diastole"]

    # Ground truth
    axes[0].plot(time, ground_truth, "g-", linewidth=0.8)
    axes[0].set_ylabel("Ground Truth")
    axes[0].set_yticks([1, 2, 3, 4])
    axes[0].set_yticklabels(labels)
    axes[0].set_ylim(0.5, 4.5)
    axes[0].grid(True, alpha=0.3)

    # Uncorrected predictions
    axes[1].plot(time, uncorrected, "r-", linewidth=0.8)
    for pos in invalid_pos:
        axes[1].axvline(x=pos, color="red", alpha=0.3, linewidth=2)
    axes[1].set_ylabel("Uncorrected")
    axes[1].set_yticks([1, 2, 3, 4])
    axes[1].set_yticklabels(labels)
    axes[1].set_ylim(0.5, 4.5)
    axes[1].set_title(f"{len(invalid_pos)} invalid transitions (red bands)")
    axes[1].grid(True, alpha=0.3)

    # Corrected predictions
    axes[2].plot(time, corrected, "b-", linewidth=0.8)
    axes[2].set_ylabel("Viterbi Corrected")
    axes[2].set_yticks([1, 2, 3, 4])
    axes[2].set_yticklabels(labels)
    axes[2].set_ylim(0.5, 4.5)
    axes[2].set_xlabel("Time step")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f"Sample {sample_idx}: Uncorrected vs Viterbi-Corrected Predictions")
    plt.tight_layout()
    plt.savefig("sample_corrections.png", dpi=150)
    print("Saved visualization to sample_corrections.png")
    plt.show()


if __name__ == "__main__":
    main()

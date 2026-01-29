#!/usr/bin/env python3
"""Find samples where the model produces invalid transitions."""

import scipy
import torch
from torchvision import transforms

from hss.datasets.heart_sounds import DavidSpringerHSS
from hss.model.lit_model import LitModel
from hss.transforms import FSST
from hss.utils.sequence_validator import CardiacCycleValidator


def main():
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

    validator = CardiacCycleValidator()

    # Search for samples with invalid transitions (stop after finding 10)
    print("Searching for samples with invalid transitions...")
    samples_with_errors = []

    for idx in range(len(dataset)):
        x, y = dataset[idx]
        x = x.unsqueeze(0)

        with torch.no_grad():
            logits = model(x).permute((0, 2, 1))
            uncorrected = torch.argmax(logits, dim=1).squeeze().numpy() + 1  # 1-indexed

        _, invalid_pos = validator.validate_sequence(uncorrected)

        if len(invalid_pos) > 0:
            samples_with_errors.append((idx, len(invalid_pos), invalid_pos[:5]))
            print(f"  Sample {idx}: {len(invalid_pos)} invalid transitions at positions {invalid_pos[:5]}...")
            if len(samples_with_errors) >= 10:
                break

    print(f"\nFound {len(samples_with_errors)} samples with invalid transitions")


if __name__ == "__main__":
    main()

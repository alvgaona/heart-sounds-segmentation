#!/usr/bin/env python3
"""Find samples where the CRF model produces invalid transitions."""

import scipy
import torch
from torchvision import transforms

from hss.datasets.heart_sounds import DavidSpringerHSS
from hss.model.lit_model_crf import LitModelCRF
from hss.transforms import FSST
from hss.utils.sequence_validator import CardiacCycleValidator


CHECKPOINT_PATH = "lightning_logs/version_0/checkpoints/epoch=14-step=1920.ckpt"


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    model = LitModelCRF.load_from_checkpoint(
        CHECKPOINT_PATH,
        input_size=44,
        batch_size=1,
        device=device,
    )
    model.eval()
    model.to(device)

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

    # Search for samples with invalid transitions
    print("Searching for samples with invalid transitions...")
    samples_with_errors = []

    for idx in range(len(dataset)):
        x, y = dataset[idx]
        x = x.unsqueeze(0).to(device)

        with torch.no_grad():
            # CRF decode returns tensor (batch, seq_len) with 0-indexed tags
            decoded = model.model.decode(model.model._get_emissions(x))
            predictions = decoded.squeeze().cpu().numpy() + 1  # Convert to 1-indexed

        _, invalid_pos = validator.validate_sequence(predictions)

        if len(invalid_pos) > 0:
            samples_with_errors.append((idx, len(invalid_pos), invalid_pos[:5]))
            print(f"  Sample {idx}: {len(invalid_pos)} invalid transitions at positions {invalid_pos[:5]}...")
            if len(samples_with_errors) >= 10:
                break

    print(f"\nFound {len(samples_with_errors)} samples with invalid transitions")


if __name__ == "__main__":
    main()

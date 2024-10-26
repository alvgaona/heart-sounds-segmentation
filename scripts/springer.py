import math
import time

import scipy
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms

from hss.datasets.heart_sounds import DavidSpringerHSS
from hss.model.segmenter import HSSegmenter
from hss.transforms import FSST
from hss.utils.training import ProgressTracker


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        (
            FSST(
                1000,
                window=scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=False),
                truncate_freq=(25, 200),
                stack=True,
            ),
        )
    )

    # Create full dataset
    hss_dataset = DavidSpringerHSS(
        "resources/data",
        download=True,
        framing=True,
        in_memory=True,
        transform=transform,
    )

    # Split into train, val, eval sets
    total_size = len(hss_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    eval_size = total_size - train_size - val_size

    train_dataset, val_dataset, eval_dataset = torch.utils.data.random_split(
        hss_dataset, [train_size, val_size, eval_size], generator=torch.Generator(device="cpu")
    )

    batch_size = 1

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator(device="cpu"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator(device="cpu"),
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator(device="cpu"),
    )

    model = HSSegmenter(input_size=44, batch_size=batch_size, device=device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = LambdaLR(optimizer, lr_lambda=[lambda epoch: 0.9**epoch])

    progress_tracker = ProgressTracker()

    mini_batch_size = 5
    start_time = time.time()
    for epoch in range(1, 7):
        model.train()
        running_loss = 0.0
        total = 0.0
        correct = 0.0

        # Training loop
        for i, (x, y) in enumerate(train_loader, 1):
            x = x.cuda().to(torch.float32)
            y = y.cuda()
            iteration = i + (epoch - 1) * math.ceil(len(train_dataset) / batch_size)

            optimizer.zero_grad()
            outputs = model(x)

            loss = loss_fn(outputs.permute((0, 2, 1)), y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = torch.max(outputs, dim=2).indices
            total += y.size(0) * y.size(1)
            correct += torch.sum(predicted == y).item()

            if iteration % mini_batch_size == 0:
                progress_tracker.show_progress(
                    epoch=epoch,
                    iteration=iteration,
                    time_elapsed=time.time() - start_time,
                    mini_batch_size=mini_batch_size,
                    mini_batch_acc=correct / total,
                    mini_batch_loss=running_loss / mini_batch_size,
                    learning_rate=scheduler.get_last_lr()[0],
                )
                running_loss = 0.0
                total = 0
                correct = 0

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_total = 0.0
        val_correct = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.cuda().to(torch.float32)
                y = y.cuda()

                outputs = model(x)
                loss = loss_fn(outputs.permute((0, 2, 1)), y)

                val_loss += loss.item()
                predicted = torch.max(outputs, dim=2).indices
                val_total += y.size(0) * y.size(1)
                val_correct += torch.sum(predicted == y).item()

        val_accuracy = val_correct / val_total
        val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch} Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        scheduler.step()


if __name__ == "__main__":
    main()

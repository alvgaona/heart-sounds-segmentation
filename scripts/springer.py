import math
import os
import time

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms

from hss.datasets.heart_sounds import DavidSpringerHSS
from hss.model.segmenter import HSSegmenter
from hss.transforms import FSST, Resample
from hss.utils.training import show_progress


ROOT = os.path.dirname(os.path.dirname(__file__))

if __name__ == "__main__":
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        (
            Resample(35500),
            # FSST(1000, window=scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=True)),
        )
    )

    hss_dataset = DavidSpringerHSS(os.path.join(ROOT, "resources/data"), download=True, transform=transform)
    batch_size = 5
    hss_loader = DataLoader(hss_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device="cuda"))

    model = HSSegmenter(batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()

    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = LambdaLR(optimizer, lr_lambda=[lambda epoch: 0.9**epoch])

    mini_batch_size = 1

    print("Training starting...")

    start_time = time.time()
    for epoch in range(1, 11):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for i, (x, y) in enumerate(hss_loader, 1):
            iteration = i + (epoch - 1) * math.ceil(len(hss_dataset) / batch_size)
            partial_correct = 0
            partial_total = 0

            optimizer.zero_grad()
            optimizer.zero_grad()
            outputs = model(x)

            loss = criterion(outputs.permute((0, 2, 1)), y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            predicted = torch.max(outputs, dim=2).indices
            partial_total += y.size(1) * y.size(0)
            partial_correct += torch.sum(predicted == y).item()
            total += partial_total
            correct += partial_correct
            running_loss += loss.item()

            # print(f"Iteration: {iteration}, Loss: {loss.item()}, Acc: {partial_correct / partial_total}")

            if iteration % mini_batch_size == mini_batch_size - 1:
                show_progress(
                    epoch=epoch,
                    iteration=iteration,
                    time_elapsed=time.time() - start_time,
                    mini_batch_size=mini_batch_size,
                    mini_batch_acc=correct / total,
                    mini_batch_loss=running_loss / mini_batch_size,
                    learning_rate=learning_rate,
                )
                running_loss = 0.0
                total = 0
                correct = 0

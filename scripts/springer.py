import datetime
import math
import os
import time

import scipy
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        (
            Resample(35500),
            FSST(
                1000,
                window=scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=True),
                truncate_freq=(25, 150),
                stack=True,
            ),
        )
    )

    hss_dataset = DavidSpringerHSS(
        os.path.join(ROOT, "resources/data"), download=True, framing=True, in_memory=True, transform=transform
    )
    batch_size = 54
    hss_loader = DataLoader(hss_dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator(device="cpu"))

    model = HSSegmenter(input_size=128, batch_size=batch_size, device=device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = LambdaLR(optimizer, lr_lambda=[lambda epoch: 0.9**epoch])

    mini_batch_size = 50
    start_time = time.time()
    for epoch in range(1, 7):
        model.train()
        running_loss = 0.0
        total = 0.0
        correct = 0.0
        for i, (x, y) in enumerate(hss_loader, 1):
            x = x.cuda()
            y = y.cuda()
            iteration = i + (epoch - 1) * math.ceil(len(hss_dataset) / batch_size)
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
                show_progress(
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
        scheduler.step()

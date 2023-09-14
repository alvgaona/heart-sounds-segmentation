import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import scipy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from hss.datasets.heart_sounds import DavidSpringerHSS
from hss.model.segmenter import HSSegmenter
from hss.transforms import FSST, Resample
from hss.utils.preprocess import frame_signal
from hss.utils.training import show_progress


ROOT = os.path.dirname(os.path.dirname(__file__))

if __name__ == "__main__":
    transform = transforms.Compose(
        (
            Resample(35500),
            # FSST(1000, yndow=scipy.signal.get_yndow(('kaiser', 0.5), 128, fftbins=True))
        )
    )

    hss_dataset = DavidSpringerHSS(
        os.path.join(ROOT, "resources/data"), download=True, transform=transform
    )
    hss_loader = DataLoader(hss_dataset, batch_size=1, shuffle=True)

    model = HSSegmenter()

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    mini_batch_size = 20

    start_time = time.time()
    for epoch in range(6):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for i, (x, y) in enumerate(hss_loader):
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)

            partial_correct = 0
            partial_total = 0

            x = x.type(torch.float32).cuda()
            optimizer.zero_grad()
            y = y.squeeze(1).cuda() - 1
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            predicted = torch.max(outputs, dim=1).indices

            partial_total += y.size(0)
            partial_correct += torch.sum(predicted == y).item()

            total += partial_total
            correct += partial_correct
            running_loss += loss.item()

            print(f"Iteration: {i + 1} Acc: {partial_correct / partial_total}", end="")
            print("", end="\r")

            if i % mini_batch_size == mini_batch_size - 1:
                show_progress(
                    epoch=epoch + 1,
                    iteration=i + 1,
                    time_elapsed=time.time() - start_time,
                    mini_batch_size=mini_batch_size,
                    mini_batch_acc=correct / total,
                    mini_batch_loss=running_loss / mini_batch_size,
                    learning_rate=learning_rate,
                )
                start_time = time.time()
                running_loss = 0.0
                total = 0
                correct = 0

import os
import time

import scipy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from hss.datasets.heart_sounds import DavidSpringerHSS
from hss.model.segmenter import HSSegmenter
from hss.transforms import FSST, Resample
from hss.utils.training import show_progress

from torch.optim.lr_scheduler import LambdaLR


ROOT = os.path.dirname(os.path.dirname(__file__))

if __name__ == "__main__":
    transform = transforms.Compose(
        (
            Resample(35500),
            FSST(1000, window=scipy.signal.get_window(('kaiser', 0.5), 128, fftbins=True))
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

    scheduler = LambdaLR(optimizer, lr_lambda=[lambda epoch: 0.9 ** epoch])

    mini_batch_size = 50

    print("Training starting...")

    start_time = time.time()
    for epoch in range(1, 11):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for i, (x, y) in enumerate(hss_loader, 1):
            iteration = i + epoch * len(hss_dataset)
            x, s, _, f = x
            f = f.squeeze(0)
            s = torch.t(s.squeeze(0))
            indices = torch.logical_and(f >= 5, f <= 200)
            f = f[indices]

            z = torch.zeros((s.shape[0], f.shape[0]), dtype=torch.complex64)

            for j in range(len(f)):
                z[:, j] = s[:, j]

            z = torch.hstack((z.real, z.imag)).cuda()
            y = y.reshape(-1, 1)

            partial_correct = 0
            partial_total = 0

            optimizer.zero_grad()
            y = y.squeeze(1).cuda() - 1
            optimizer.zero_grad()
            outputs = model(z)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            predicted = torch.max(outputs, dim=1).indices

            partial_total += y.size(0)
            partial_correct += torch.sum(predicted == y).item()
            total += partial_total
            correct += partial_correct
            running_loss += loss.item()

            # print(f"Loss: {loss.item()}, Acc: {partial_correct / partial_total}")

            if i % mini_batch_size == mini_batch_size - 1:
                show_progress(
                    epoch=epoch + 1,
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

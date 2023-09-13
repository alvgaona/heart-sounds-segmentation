import os

import torch
from torch.utils.data import DataLoader

from hss.datasets.heart_sounds import DavidSpringerHSS


ROOT = os.path.dirname(os.path.dirname(__file__))

if __name__ == "__main__":
    hss_dataset = DavidSpringerHSS(os.path.join(ROOT, "resources/data"), download=True)
    hss_loader = DataLoader(hss_dataset, batch_size=1, shuffle=True)

    mean = 0
    std = 0
    lengths = []
    for k, v in enumerate(hss_loader):
        x, y = v
        mean += torch.mean(x)
        std += torch.std(x, unbiased=True)
        lengths.append(x.shape[1])
    print(mean / len(hss_dataset))
    print(std / len(hss_dataset))
    print(max(lengths))

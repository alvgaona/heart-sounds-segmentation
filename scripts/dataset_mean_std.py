import torch
from torch.utils.data import DataLoader

from hss.datasets.heart_sounds import DavidSpringerHSS


if __name__ == "__main__":
    hss_dataset = DavidSpringerHSS("resources/data", download=True)
    hss_loader = DataLoader(hss_dataset, batch_size=1, shuffle=True)

    mean = 0.0
    std = 0.0
    lengths = []
    for _, v in enumerate(hss_loader):
        x, y = v
        mean += torch.mean(x).item()
        std += torch.std(x, unbiased=True).item()
        lengths.append(x.shape[1])
    print(mean / len(hss_dataset))
    print(std / len(hss_dataset))
    print(max(lengths))

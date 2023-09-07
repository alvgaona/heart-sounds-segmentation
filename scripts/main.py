import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, Normalize

from hss.datasets.heart_sounds import PhysionetChallenge2016
from hss.transforms import FSST, Resample

if __name__ == '__main__':
    root = './'

    transform = transforms.Compose([
        Resample(250000),
        FSST(1000, window=scipy.signal.get_window(('kaiser', 0.5), 128, fftbins=True)),
    ])

    train_hs_dataset = PhysionetChallenge2016(root, train=True, download=True, transform=transform)
    val_hs_dataset = PhysionetChallenge2016(root, train=False, download=True, transform=transform)

    train_hs_loader = DataLoader(train_hs_dataset, batch_size=1, shuffle=True)
    val_hs_loader = DataLoader(val_hs_dataset, batch_size=1, shuffle=True)

    samples = torch.tensor(list(range(10)), dtype=torch.float64)
    print(torch.mean(samples))
    print(torch.std(samples, unbiased=False))

    mu = 0.0
    std = 0.0

    for k, x in enumerate(train_hs_loader, 1):
        s, _, f = x[0]
        s, f = s[0], f[0]

        f_indices = np.logical_and(f > 5, f < 50).to(torch.bool)
        f = f[f_indices]

        s = s[f_indices, :]

        sbar = torch.cat((torch.real(s), torch.imag(s)))

        sbar_mu = torch.mean(sbar)
        sbar_std = torch.std(sbar, unbiased=False)

        old_mu = mu
        mu += (sbar_mu - mu) / k
        std += (sbar_mu - mu) * (sbar_mu - old_mu)

    std /= (len(samples) - 1)
    std = np.sqrt(std)

    print(mu, std)
    # print(torch.mean(means))
    # print(torch.std(stds))

    # plt.figure(1)
    # plt.title('Synchrosqueezed Short-time Fourier Transform')
    # plt.imshow(np.abs(Tsx), cmap='jet', aspect='auto', vmin=0, vmax=1)
    # plt.xlabel('n [samples]')

    # plt.figure(3)
    # plt.title('Short-time Fourier Transform')
    # plt.imshow(np.abs(Sx), cmap='jet', aspect='auto', vmin=0, vmax=1)
    # plt.xlabel('n [samples]')

    # plt.show()

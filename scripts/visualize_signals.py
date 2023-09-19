import os

import matplotlib.pyplot as plt
import pandas as pd
import scipy
import torch
from torchvision import transforms

from hss.transforms import FSST, Resample


ROOT = os.path.dirname(os.path.dirname(__file__))


if __name__ == "__main__":
    df = pd.read_csv("../resources/data/springer_sounds/0040.csv", skiprows=1, names=["Signals", "Labels"])
    x = torch.tensor(df.loc[:, "Signals"].to_numpy())
    y = torch.tensor(df.loc[:, "Labels"].to_numpy(), dtype=torch.int64)

    plt.figure("PCG")
    plt.title("PCG Signal")
    plt.plot(x)
    plt.plot(y)
    plt.xlabel("n [samples]")
    plt.ylabel("Amplitude")
    plt.show()

    transform = transforms.Compose(
        (
            Resample(35500),
            FSST(
                4000,
                window=scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=True),
                truncate_freq=(25, 120),
                stack=False,
                flipud=False,
            ),
        )
    )

    s = transform(x).cpu().squeeze(0)

    plt.figure("FSST")
    plt.title("Synchrosqueezed Short-time Fourier Transform")
    plt.imshow(torch.abs(s), cmap="jet", aspect="auto", vmin=0, vmax=1, origin="lower")
    plt.xlabel("n [samples]")
    plt.ylabel("frequencies")
    plt.show()

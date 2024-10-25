import pandas as pd
import scipy
import torch
from torchvision import transforms

from hss.transforms import FSST, Resample


if __name__ == "__main__":
    df = pd.read_csv("./resources/data/springer_sounds/0001.csv", skiprows=1, names=["Signals", "Labels"])
    x = torch.tensor(df.loc[:, "Signals"].to_numpy())
    y = torch.tensor(df.loc[:, "Labels"].to_numpy(), dtype=torch.int64)

    # plt.figure("PCG")
    # plt.title("PCG Signal")
    # plt.plot(x)
    # plt.plot(y)
    # plt.xlabel("n [samples]")
    # plt.ylabel("Amplitude")

    transform = transforms.Compose(
        (
            Resample(35500),
            FSST(
                1000,
                window=scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=False),
                stack=False,
            ),
        )
    )

    s = transform(x).cpu().squeeze(0)

    print(s.shape)
    print(s)

    # plt.figure("FSST")
    # plt.title("Synchrosqueezed Short-time Fourier Transform")
    # plt.imshow(torch.abs(s), cmap="jet", aspect="auto", vmin=0, vmax=1, origin="lower")
    # plt.xlabel("n [samples]")
    # plt.ylabel("frequencies")
    # plt.show()

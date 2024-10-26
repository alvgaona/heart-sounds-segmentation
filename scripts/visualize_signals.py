import pandas as pd
import scipy
import torch
from torchvision import transforms

from hss.transforms.synchrosqueeze import FSST
from hss.utils.preprocess import frame_signal


if __name__ == "__main__":
    df = pd.read_csv("./resources/data/springer_sounds/0001.csv", skiprows=1, names=["Signals", "Labels"])
    x = torch.tensor(df.loc[:, "Signals"].to_numpy())
    y = torch.tensor(df.loc[:, "Labels"].to_numpy(), dtype=torch.int64)

    frames, labels = frame_signal(x, y, 1000, 2000)

    transform = transforms.Compose(
        (
            FSST(
                1000,
                window=scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=False),
                stack=True,
                truncate_freq=(25, 200),
            ),
        )
    )

    s = transform(frames[0]).cpu().squeeze(0)

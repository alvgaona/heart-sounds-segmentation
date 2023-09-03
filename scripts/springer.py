import os

import pandas as pd
import plotly.express as px
import scipy
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from hss.datasets.heart_sounds import DavidSpringerHSS
from hss.transforms import FSST, Resample


ROOT = os.path.dirname(os.path.dirname(__file__))


if __name__ == '__main__':
    resample = Resample(250000)

    transform = transforms.Compose((
        resample,
        FSST(1000, window=scipy.signal.get_window(('kaiser', 0.5), 128, fftbins=True))
    ))

    hss_dataset = DavidSpringerHSS(os.path.join(ROOT, "resources/data"), download=True, transform=transform)
    hss_loader = DataLoader(hss_dataset, batch_size=1, shuffle=True)

    x, y = next(iter(hss_loader))
    x, *_ = x
    df = pd.DataFrame(torch.hstack((x.reshape(-1, 1), y.reshape(-1, 1))))
    fig = px.line(df, title='PCG')
    fig.show()

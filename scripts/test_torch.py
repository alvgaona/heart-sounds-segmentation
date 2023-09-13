import torch
from torch import nn


if __name__ == "__main__":
    h0, c0 = (
        torch.randn(1, 240),
        torch.randn(1, 240),
    )

    x = torch.randn((1, 35500))
    print(x.shape)
    model = nn.LSTM(input_size=35500, hidden_size=240, num_layers=1)
    output, (hn, cn) = model(x, (h0, c0))

import torch
from torch import nn


class HSSegmenter(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        batch_size: int = 1,
        hidden_size: int = 240,
        num_layers: int = 2,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.batch_size = batch_size

        D = 2 if bidirectional else 1

        self.h0, self.c0 = (
            torch.randn(D * num_layers, batch_size, hidden_size),
            torch.randn(D * num_layers, batch_size, hidden_size),
        )

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=hidden_size * 2, out_features=4, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x, (self.h0, self.c0))
        output = self.relu(output)
        output = self.linear(output)
        return self.softmax(output)

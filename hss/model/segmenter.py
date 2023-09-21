import torch
from torch import nn


class HSSegmenter(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        batch_size: int = 1,
        hidden_size: int = 240,
        bidirectional: bool = True,
        device: torch.device = torch.cpu,
    ):
        super().__init__()

        self.batch_size = batch_size

        D = 2 if bidirectional else 1

        self.h0, self.c0 = (
            torch.randn(D, batch_size, hidden_size, device=device),
            torch.randn(D, batch_size, hidden_size, device=device),
        )

        self.lstm_1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            batch_first=True,
            device=device,
        )
        self.lstm_2 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            batch_first=True,
            device=device,
        )
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=hidden_size * 2, out_features=4, bias=True, device=device)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, (hn, cn) = self.lstm_1(x, (self.h0, self.c0))
        output = self.dropout(output)
        output = self.relu(output)
        output, _ = self.lstm_2(output, (hn, cn))
        output = self.dropout(output)
        output = self.relu(output)
        output = self.linear(output)
        return self.softmax(output)

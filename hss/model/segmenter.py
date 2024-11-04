import torch
from torch import nn


class HeartSoundSegmenter(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int = 240,
        bidirectional: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.device = device if device else torch.device("cpu")
        self.dtype = dtype

        self.lstm_1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            batch_first=True,
            device=device,
            dtype=dtype,
        )

        self.lstm_2 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            batch_first=True,
            device=device,
            dtype=dtype,
        )

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=hidden_size * 2, out_features=4, bias=True, device=device, dtype=dtype)
        self.softmax = nn.LogSoftmax(dim=2)

    def _init_hidden(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        D = 2 if self.bidirectional else 1
        h0 = torch.randn(
            D,
            batch_size,
            self.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )
        c0 = torch.randn(
            D,
            batch_size,
            self.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )
        return h0, c0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0, c0 = self._init_hidden(x.size(0))

        output, (hn, cn) = self.lstm_1(x, (h0, c0))

        output = self.relu(output)
        output = self.dropout(output)

        output, _ = self.lstm_2(output, (hn, cn))
        output = self.relu(output)
        output = self.dropout(output)

        output = self.linear(output)
        return self.softmax(output)

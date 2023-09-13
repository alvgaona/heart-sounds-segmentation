import torch
from torch import nn


class HSSegmenter(nn.Module):
    def __init__(self):
        super().__init__()

        self.hn, self.cn = (
            torch.randn(4, 240).cuda(),
            torch.randn(4, 240).cuda(),
        )

        self.lstm_1 = nn.LSTM(
            input_size=1, hidden_size=240, num_layers=2, bidirectional=True, dropout=0.2
        ).cuda()
        self.relu = nn.ReLU().cuda()
        self.linear = nn.Linear(in_features=480, out_features=4).cuda()
        self.softmax = nn.Softmax(dim=1).cuda()

    def forward(self, x):
        output, (self.hn, self.cn) = self.lstm_1(
            x, (self.hn.detach(), self.cn.detach())
        )
        output = self.relu(output)
        output = self.linear(output)
        return self.softmax(output)

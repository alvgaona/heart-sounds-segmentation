import torch
from torch import nn


class HSSegmenter(nn.Module):
    def __init__(self):
        super().__init__()

        self.hn1, self.cn1 = (
            torch.randn(2, 240).cuda(),
            torch.randn(2, 240).cuda(),
        )

        self.hn2, self.cn2 = (
            torch.randn(2, 480).cuda(),
            torch.randn(2, 480).cuda(),
        )

        self.dropout = nn.Dropout(0.2)
        self.lstm_1 = nn.LSTM(input_size=200, hidden_size=240, num_layers=1, bidirectional=True).cuda()
        self.lstm_2 = nn.LSTM(input_size=480, hidden_size=480, num_layers=1, bidirectional=True).cuda()
        self.relu = nn.ReLU().cuda()
        self.linear = nn.Linear(in_features=960, out_features=4).cuda()
        self.softmax = nn.Softmax(dim=1).cuda()

    def forward(self, x):
        output, (self.hn1, self.cn1) = self.lstm_1(x, (self.hn1.detach(), self.cn1.detach()))
        output = self.dropout(output)
        output = self.relu(output)
        output, (self.hn2, self.cn2) = self.lstm_2(output, (self.hn2.detach(), self.cn2.detach()))
        output = self.relu(output)
        output = self.linear(output)
        return self.softmax(output)

"""Heart sound segmenter with attention mechanism."""

import torch
from torch import nn


class HeartSoundSegmenterAttention(nn.Module):
    """Neural network model for segmenting heart sounds with self-attention.

    This model uses a two-layer bidirectional LSTM followed by multi-head
    self-attention to capture long-range dependencies in the sequence.

    Args:
        input_size: Size of input features at each time step
        batch_size: Number of sequences in each batch
        hidden_size: Number of hidden units in each LSTM layer
        num_attention_heads: Number of attention heads
        bidirectional: Whether to use bidirectional LSTMs
        device: Device to place the model on (CPU/GPU)
        dtype: Data type for model parameters
    """

    def __init__(
        self,
        *,
        input_size: int,
        batch_size: int = 1,
        hidden_size: int = 240,
        num_attention_heads: int = 8,
        bidirectional: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.device = device if device is not None else torch.device("cpu")
        self.batch_size = batch_size

        D = 2 if bidirectional else 1
        lstm_output_size = hidden_size * D

        self.h0, self.c0 = (
            torch.randn(D, batch_size, hidden_size, device=self.device, dtype=dtype),
            torch.randn(D, batch_size, hidden_size, device=self.device, dtype=dtype),
        )

        self.lstm_1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            batch_first=True,
            device=self.device,
            dtype=dtype,
        )
        self.lstm_2 = nn.LSTM(
            input_size=lstm_output_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            batch_first=True,
            device=self.device,
            dtype=dtype,
        )

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=num_attention_heads,
            dropout=0.1,
            batch_first=True,
            device=self.device,
            dtype=dtype,
        )
        self.attention_norm = nn.LayerNorm(lstm_output_size, device=self.device, dtype=dtype)

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

        self.linear = nn.Linear(
            in_features=lstm_output_size,
            out_features=4,
            bias=True,
            device=self.device,
            dtype=dtype,
        )
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Log softmax probabilities for each class at each time step,
            shape (batch_size, sequence_length, 4)
        """
        # LSTM layers
        output, (hn, cn) = self.lstm_1(x, (self.h0, self.c0))
        output = self.relu(output)
        output = self.dropout(output)

        output, _ = self.lstm_2(output, (hn, cn))
        output = self.relu(output)
        output = self.dropout(output)

        # Self-attention with residual connection
        attn_output, _ = self.attention(output, output, output)
        output = self.attention_norm(output + attn_output)
        output = self.dropout(output)

        # Classification
        output = self.linear(output)
        return self.softmax(output)

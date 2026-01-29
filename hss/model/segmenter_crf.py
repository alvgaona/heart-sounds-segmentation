"""Heart sound segmenter with CRF layer for sequence modeling."""

import torch
from torch import nn
from torchcrf import CRF


class HeartSoundSegmenterCRF(nn.Module):
    """Neural network model for segmenting heart sounds with CRF sequence modeling.

    This model uses a two-layer bidirectional LSTM followed by a CRF layer
    to jointly model transitions between cardiac states.

    The CRF layer learns transition scores and enforces sequence constraints
    during both training and inference.

    Args:
        input_size: Size of input features at each time step
        batch_size: Number of sequences in each batch
        hidden_size: Number of hidden units in each LSTM layer
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
        bidirectional: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.device = device if device is not None else torch.device("cpu")
        self.batch_size = batch_size
        self.num_tags = 4  # S1, Systole, S2, Diastole

        D = 2 if bidirectional else 1

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
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            batch_first=True,
            device=self.device,
            dtype=dtype,
        )
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(
            in_features=hidden_size * 2,
            out_features=self.num_tags,
            bias=True,
            device=self.device,
            dtype=dtype,
        )

        # CRF layer for sequence modeling
        self.crf = CRF(num_tags=self.num_tags, batch_first=True)

        # Initialize CRF transitions with cardiac cycle constraints
        self._init_crf_transitions()

    def _init_crf_transitions(self) -> None:
        """Initialize CRF transition matrix with cardiac cycle prior.

        Valid transitions: 0->0, 0->1, 1->1, 1->2, 2->2, 2->3, 3->3, 3->0
        Invalid transitions get negative initialization to discourage them.
        """
        with torch.no_grad():
            # Start with small negative values for all transitions
            self.crf.transitions.fill_(-1.0)

            # Set valid transitions to positive values
            # Self-transitions (staying in same state)
            self.crf.transitions[0, 0] = 1.0  # S1 -> S1
            self.crf.transitions[1, 1] = 1.0  # Systole -> Systole
            self.crf.transitions[2, 2] = 1.0  # S2 -> S2
            self.crf.transitions[3, 3] = 1.0  # Diastole -> Diastole

            # Forward transitions (cardiac cycle order)
            self.crf.transitions[1, 0] = 1.0  # S1 -> Systole (0 -> 1)
            self.crf.transitions[2, 1] = 1.0  # Systole -> S2 (1 -> 2)
            self.crf.transitions[3, 2] = 1.0  # S2 -> Diastole (2 -> 3)
            self.crf.transitions[0, 3] = 1.0  # Diastole -> S1 (3 -> 0)

    def _get_emissions(self, x: torch.Tensor) -> torch.Tensor:
        """Compute emission scores from input.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Emission scores of shape (batch_size, sequence_length, num_tags)
        """
        output, (hn, cn) = self.lstm_1(x, (self.h0, self.c0))
        output = self.relu(output)
        output = self.dropout(output)
        output, _ = self.lstm_2(output, (hn, cn))
        output = self.relu(output)
        output = self.dropout(output)
        return self.linear(output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning emission scores.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Emission scores of shape (batch_size, sequence_length, num_tags)
        """
        return self._get_emissions(x)

    def loss(self, x: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        """Compute negative log-likelihood loss.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            tags: Target tags of shape (batch_size, sequence_length)

        Returns:
            Negative log-likelihood loss (scalar)
        """
        emissions = self._get_emissions(x)
        return -self.crf(emissions, tags)

    def decode(self, x: torch.Tensor) -> list[list[int]]:
        """Decode the best tag sequence using Viterbi algorithm.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            List of best tag sequences for each sample in batch
        """
        emissions = self._get_emissions(x)
        return self.crf.decode(emissions)

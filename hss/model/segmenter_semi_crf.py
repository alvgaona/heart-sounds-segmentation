"""Heart sound segmenter with Semi-Markov CRF for duration-aware sequence modeling."""

import torch
from torch import nn

from hss.model.semi_markov_crf import SemiMarkovCRF


class HeartSoundSegmenterSemiCRF(nn.Module):
    """Neural network model for segmenting heart sounds with Semi-Markov CRF.

    This model uses a two-layer bidirectional LSTM followed by a Semi-Markov CRF layer
    that explicitly models segment durations in addition to state transitions.

    Unlike the standard CRF which models frame-by-frame transitions, the Semi-Markov CRF
    models entire segments with learnable duration distributions per state. This is similar
    to Springer's Hidden Semi-Markov Model (HSMM) approach.

    Args:
        input_size: Size of input features at each time step
        batch_size: Number of sequences in each batch
        hidden_size: Number of hidden units in each LSTM layer
        bidirectional: Whether to use bidirectional LSTMs
        max_duration: Maximum segment duration in frames
        duration_means: Initial mean duration for each state (in frames)
        duration_stds: Initial std duration for each state (in frames)
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
        max_duration: int = 500,
        duration_means: list[float] | None = None,
        duration_stds: list[float] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.device = device if device is not None else torch.device("cpu")
        self.batch_size = batch_size
        self.num_tags = 4  # S1, Systole, S2, Diastole

        D = 2 if bidirectional else 1

        self.register_buffer("h0", torch.randn(D, batch_size, hidden_size, device=self.device, dtype=dtype))
        self.register_buffer("c0", torch.randn(D, batch_size, hidden_size, device=self.device, dtype=dtype))

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

        # Semi-Markov CRF layer with duration modeling
        self.crf = SemiMarkovCRF(
            num_tags=self.num_tags,
            max_duration=max_duration,
            duration_means=duration_means,
            duration_stds=duration_stds,
        )

    def _get_emissions(self, x: torch.Tensor) -> torch.Tensor:
        """Compute emission scores from input.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Emission scores of shape (batch_size, sequence_length, num_tags)
        """
        h0 = self.h0[:, : x.shape[0], :]
        c0 = self.c0[:, : x.shape[0], :]

        out, _ = self.lstm_1(x, (h0, c0))
        out = self.dropout(out)
        out = self.relu(out)
        out, _ = self.lstm_2(out, (h0, c0))
        out = self.dropout(out)
        out = self.relu(out)

        emissions = self.linear(out)
        return emissions

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
            tags: Ground truth tags of shape (batch_size, sequence_length)

        Returns:
            Negative log-likelihood loss (scalar)
        """
        emissions = self._get_emissions(x)
        return self.crf(emissions, tags)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode the best tag sequence using Semi-Markov Viterbi algorithm.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Best tag sequences (batch_size, sequence_length)
        """
        emissions = self._get_emissions(x)
        return self.crf.decode(emissions)

    def decode_segments(self, x: torch.Tensor) -> list[list[tuple[int, int, int]]]:
        """Decode the best segmentation.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            List of segment lists per batch item, each segment is (state, start, end)
        """
        emissions = self._get_emissions(x)
        return self.crf.decode_segments(emissions)

    def get_duration_params(self) -> dict[str, torch.Tensor]:
        """Get learned duration parameters.

        Returns:
            Dictionary with 'means' and 'stds' tensors
        """
        return {
            "means": self.crf.duration_means.detach(),
            "stds": self.crf.duration_stds.detach(),
        }

    def marginals(self, x: torch.Tensor) -> torch.Tensor:
        """Compute marginal probabilities P(y_t = k | x) using forward-backward.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Marginal probabilities (batch_size, sequence_length, num_tags)
        """
        emissions = self._get_emissions(x)
        return self.crf.marginals(emissions)

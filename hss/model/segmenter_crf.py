"""Heart sound segmenter with CRF layer for sequence modeling."""

import torch
from torch import nn

from hss.model.crf import CRF
from hss.model.xlstm import _pool_time, _upsample_time
from hss.utils.sequence_validator import validate_and_correct_predictions


class MultiRateBiLSTMEncoder(nn.Module):
    """Temporal pyramid of bidirectional LSTMs at several rates (Experiment D, BiLSTM base).

    Mirror of ``MultiRateXLSTMEncoder`` but with a plain ``nn.LSTM`` per rate — isolates the multi-scale
    hierarchy from the emitter (BiLSTM is the accuracy leader). Emits ``(B, T, 2 * hidden * len(rates))``.
    """

    def __init__(
        self, input_size: int, hidden_size: int = 240, num_layers: int = 1, rates: tuple[int, ...] = (1, 4, 16)
    ) -> None:
        super().__init__()
        self.rates = tuple(rates)
        self.levels = nn.ModuleList(
            nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
            for _ in self.rates
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        outs = []
        for rate, level in zip(self.rates, self.levels, strict=True):
            y, _ = level(_pool_time(x, rate))
            outs.append(_upsample_time(y, rate, t))
        return torch.cat(outs, dim=-1)


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
        multirate: bool = False,
        num_layers: int = 1,
        rates: tuple[int, ...] = (1, 4, 16),
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.device = device if device is not None else torch.device("cpu")
        self.batch_size = batch_size
        self.num_tags = 4  # S1, Systole, S2, Diastole
        self.multirate = multirate
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

        if multirate:
            # Multi-rate temporal pyramid (Experiment D) — no h0/c0; the encoder owns its LSTM state.
            self.encoder = MultiRateBiLSTMEncoder(input_size, hidden_size, num_layers, rates)
            linear_in = hidden_size * 2 * len(rates)
        else:
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
            linear_in = hidden_size * 2
        self.linear = nn.Linear(in_features=linear_in, out_features=self.num_tags, bias=True)

        # CRF layer for sequence modeling
        self.crf = CRF(num_tags=self.num_tags)

        # Initialize CRF transitions with cardiac cycle constraints
        self._init_crf_transitions()

    def _init_crf_transitions(self) -> None:
        """Initialize CRF transition matrix with cardiac cycle prior.

        Valid transitions: 0->0, 0->1, 1->1, 1->2, 2->2, 2->3, 3->3, 3->0
        Invalid transitions get negative initialization to discourage them.

        Note: transitions[i, j] = score for i -> j
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
            self.crf.transitions[0, 1] = 1.0  # S1 -> Systole (0 -> 1)
            self.crf.transitions[1, 2] = 1.0  # Systole -> S2 (1 -> 2)
            self.crf.transitions[2, 3] = 1.0  # S2 -> Diastole (2 -> 3)
            self.crf.transitions[3, 0] = 1.0  # Diastole -> S1 (3 -> 0)

            # Start transitions: likely to start with S1 or Diastole
            self.crf.start_transitions.fill_(-1.0)
            self.crf.start_transitions[0] = 1.0  # S1
            self.crf.start_transitions[3] = 0.5  # Diastole

            # End transitions: any state is valid
            self.crf.end_transitions.fill_(0.0)

    def _get_emissions(self, x: torch.Tensor) -> torch.Tensor:
        """Compute emission scores from input.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Emission scores of shape (batch_size, sequence_length, num_tags)
        """
        if self.multirate:
            return self.linear(self.dropout(self.encoder(x)))
        # Slice h0/c0 to the actual batch size so partial batches (last CV/test batch) work, and
        # .contiguous() because a sliced view is non-contiguous, which the CUDA LSTM kernel rejects.
        h0 = self.h0[:, : x.shape[0], :].contiguous()
        c0 = self.c0[:, : x.shape[0], :].contiguous()
        output, (hn, cn) = self.lstm_1(x, (h0, c0))
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
        return self.crf(emissions, tags)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode the best tag sequence using Viterbi algorithm.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Best tag sequences (batch_size, sequence_length)
        """
        emissions = self._get_emissions(x)
        return self.crf.decode(emissions)

    def marginals(self, x: torch.Tensor) -> torch.Tensor:
        """Compute marginal probabilities P(y_t = k | x) using forward-backward.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Marginal probabilities (batch_size, sequence_length, num_tags)
        """
        emissions = self._get_emissions(x)
        return self.crf.marginals(emissions)

    def decode_valid(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a guaranteed-valid cardiac cycle via constrained-posterior decoding.

        Runs a frame-level constrained Viterbi (self-loops plus S1→Systole→S2→Diastole→S1 only) over
        the forward-backward posterior marginals — the same decoder the Semi-Markov CRF uses, so all
        models can be compared under one decode method.

        Returns:
            Best valid tag sequences (batch_size, sequence_length), labels 0-3.
        """
        marginals = self.marginals(x)
        log_posterior = torch.log(marginals.clamp_min(1e-9))
        corrected = torch.as_tensor(validate_and_correct_predictions(log_posterior), device=x.device)
        return corrected - 1

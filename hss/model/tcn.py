"""Temporal Convolutional Network (TCN) for sequence modeling.

TCN uses dilated causal convolutions to achieve large receptive fields
while maintaining computational efficiency and full parallelism.

Supports both causal (unidirectional) and bidirectional modes.

References:
- Bai et al. 2018: "An Empirical Evaluation of Generic Convolutional and
  Recurrent Networks for Sequence Modeling"
"""

import torch
from torch import Tensor, nn


class CausalConv1d(nn.Module):
    """Causal 1D convolution with dilation.

    Pads the input so that the output only depends on past and current inputs,
    not future inputs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, channels, seq_len)
        out = self.conv(x)
        # Remove the extra padding from the right (future) side
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        return out


class TCNBlock(nn.Module):
    """Single TCN residual block with two causal convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)

        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.relu = nn.ReLU()

        # Residual connection (1x1 conv if dimensions don't match)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, channels, seq_len)
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        return self.relu(out + residual)


class TCN(nn.Module):
    """Temporal Convolutional Network.

    Uses stacked dilated causal convolutions with exponentially increasing
    dilation factors to achieve a large receptive field.

    Args:
        input_size: Number of input features per timestep
        hidden_size: Number of channels in hidden layers
        num_layers: Number of TCN blocks; RF = 1 + 2*(kernel_size-1)*(2^num_layers - 1)
        kernel_size: Convolution kernel size
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.kernel_size = kernel_size

        # Receptive field. Each TCNBlock has TWO dilated convs at dilation 2^i, so each block
        # contributes 2*(k-1)*2^i. Summed over layers: RF = 1 + 2*(k-1)*(2^num_layers - 1).
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2**num_layers - 1)

        # Input projection
        self.input_proj = nn.Conv1d(input_size, hidden_size, 1)

        # TCN blocks with exponentially increasing dilation
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2**i
            self.blocks.append(
                TCNBlock(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        # Transpose to (batch, channels, seq_len) for conv1d
        x = x.transpose(1, 2)

        # Project input to hidden size
        x = self.input_proj(x)

        # Apply TCN blocks
        for block in self.blocks:
            x = block(x)

        # Transpose back to (batch, seq_len, hidden_size)
        return x.transpose(1, 2)


class HeartSoundSegmenterTCN(nn.Module):
    """TCN-based heart sound segmenter with CRF.

    Uses a Temporal Convolutional Network for feature extraction followed
    by a linear CRF for sequence labeling.

    Args:
        input_size: Number of input features (44 for FSST)
        hidden_size: TCN hidden dimension
        num_layers: Number of TCN blocks
        kernel_size: Convolution kernel size
        dropout: Dropout probability
        num_tags: Number of output classes (4 for heart sounds)
    """

    def __init__(
        self,
        input_size: int = 44,
        hidden_size: int = 256,
        num_layers: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.2,
        num_tags: int = 4,
    ):
        super().__init__()

        self.tcn = TCN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.classifier = nn.Linear(hidden_size, num_tags)

        # Import CRF here to avoid circular imports
        from hss.model.crf import CRF

        self.crf = CRF(num_tags)

        # Initialize CRF transitions to encourage cardiac cycle
        self._init_crf_transitions()

    def _init_crf_transitions(self):
        """Initialize CRF transitions for frame-level cardiac-cycle labeling.

        This is a frame-level CRF: a state persists across many frames, so self-transitions
        dominate and are encouraged (positive diagonal). The only valid state change is the forward
        cardiac cycle S1 -> Systole -> S2 -> Diastole -> S1; all other (backward / skip) transitions
        are discouraged. (Note: the semi-Markov CRF forbids self-transitions because its segments are
        maximal — that convention must NOT be copied here.)
        """
        with torch.no_grad():
            # Default: discourage every transition, which suppresses the invalid (backward/skip) ones.
            self.crf.transitions.fill_(-5.0)
            # Self-transitions: a state persists across most frames.
            self.crf.transitions.fill_diagonal_(2.0)
            # Forward cardiac cycle transitions (the only valid state changes; rarer than staying).
            self.crf.transitions[0, 1] = 1.0  # S1 -> Systole
            self.crf.transitions[1, 2] = 1.0  # Systole -> S2
            self.crf.transitions[2, 3] = 1.0  # S2 -> Diastole
            self.crf.transitions[3, 0] = 1.0  # Diastole -> S1

    def forward(self, x: Tensor) -> Tensor:
        """Compute emission scores.

        Args:
            x: Input of shape (batch, seq_len, input_size)

        Returns:
            Emissions of shape (batch, seq_len, num_tags)
        """
        features = self.tcn(x)
        return self.classifier(features)

    def loss(self, x: Tensor, tags: Tensor) -> Tensor:
        """Compute CRF negative log-likelihood loss.

        Args:
            x: Input of shape (batch, seq_len, input_size)
            tags: Ground truth tags of shape (batch, seq_len)

        Returns:
            Scalar loss
        """
        emissions = self.forward(x)
        return self.crf(emissions, tags)

    def decode(self, x: Tensor) -> Tensor:
        """Decode best tag sequence using Viterbi.

        Args:
            x: Input of shape (batch, seq_len, input_size)

        Returns:
            Best tags of shape (batch, seq_len)
        """
        emissions = self.forward(x)
        return self.crf.decode(emissions)

    def marginals(self, x: Tensor) -> Tensor:
        """Compute marginal probabilities P(y_t = k | x) via the CRF forward-backward.

        Args:
            x: Input of shape (batch, seq_len, input_size)

        Returns:
            Marginal probabilities of shape (batch, seq_len, num_tags)
        """
        emissions = self.forward(x)
        return self.crf.marginals(emissions)


class BiTCN(nn.Module):
    """Bidirectional Temporal Convolutional Network.

    Runs two TCNs in parallel:
    - Forward TCN: processes sequence left-to-right (causal)
    - Backward TCN: processes sequence right-to-left (anti-causal)

    Outputs are concatenated, similar to BiLSTM.

    Args:
        input_size: Number of input features per timestep
        hidden_size: Number of channels in each direction (total output = 2 * hidden_size)
        num_layers: Number of TCN blocks per direction
        kernel_size: Convolution kernel size
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size * 2  # Concatenated forward + backward

        # Forward (causal) TCN
        self.forward_tcn = TCN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # Backward (anti-causal) TCN
        self.backward_tcn = TCN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.receptive_field = self.forward_tcn.receptive_field

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size * 2)
        """
        # Forward direction: normal processing
        forward_out = self.forward_tcn(x)  # (batch, seq_len, hidden_size)

        # Backward direction: flip, process, flip back
        x_reversed = x.flip(dims=[1])
        backward_out = self.backward_tcn(x_reversed)
        backward_out = backward_out.flip(dims=[1])  # (batch, seq_len, hidden_size)

        # Concatenate forward and backward
        return torch.cat([forward_out, backward_out], dim=-1)


class HeartSoundSegmenterBiTCN(nn.Module):
    """Bidirectional TCN-based heart sound segmenter with CRF.

    Uses a Bidirectional Temporal Convolutional Network for feature extraction
    followed by a linear CRF for sequence labeling. Similar to BiLSTM + CRF
    but fully parallelizable.

    Args:
        input_size: Number of input features (44 for FSST)
        hidden_size: TCN hidden dimension per direction (total = 2 * hidden_size)
        num_layers: Number of TCN blocks per direction
        kernel_size: Convolution kernel size
        dropout: Dropout probability
        num_tags: Number of output classes (4 for heart sounds)
    """

    def __init__(
        self,
        input_size: int = 44,
        hidden_size: int = 256,
        num_layers: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.2,
        num_tags: int = 4,
    ):
        super().__init__()

        self.bitcn = BiTCN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # Classifier takes concatenated forward + backward features
        self.classifier = nn.Linear(hidden_size * 2, num_tags)

        # Import CRF here to avoid circular imports
        from hss.model.crf import CRF

        self.crf = CRF(num_tags)

        # Initialize CRF transitions to encourage cardiac cycle
        self._init_crf_transitions()

    def _init_crf_transitions(self):
        """Initialize CRF transitions for frame-level cardiac-cycle labeling.

        This is a frame-level CRF: a state persists across many frames, so self-transitions
        dominate and are encouraged (positive diagonal). The only valid state change is the forward
        cardiac cycle S1 -> Systole -> S2 -> Diastole -> S1; all other (backward / skip) transitions
        are discouraged. (Note: the semi-Markov CRF forbids self-transitions because its segments are
        maximal — that convention must NOT be copied here.)
        """
        with torch.no_grad():
            # Default: discourage every transition, which suppresses the invalid (backward/skip) ones.
            self.crf.transitions.fill_(-5.0)
            # Self-transitions: a state persists across most frames.
            self.crf.transitions.fill_diagonal_(2.0)
            # Forward cardiac cycle transitions (the only valid state changes; rarer than staying).
            self.crf.transitions[0, 1] = 1.0  # S1 -> Systole
            self.crf.transitions[1, 2] = 1.0  # Systole -> S2
            self.crf.transitions[2, 3] = 1.0  # S2 -> Diastole
            self.crf.transitions[3, 0] = 1.0  # Diastole -> S1

    def forward(self, x: Tensor) -> Tensor:
        """Compute emission scores.

        Args:
            x: Input of shape (batch, seq_len, input_size)

        Returns:
            Emissions of shape (batch, seq_len, num_tags)
        """
        features = self.bitcn(x)
        return self.classifier(features)

    def loss(self, x: Tensor, tags: Tensor) -> Tensor:
        """Compute CRF negative log-likelihood loss.

        Args:
            x: Input of shape (batch, seq_len, input_size)
            tags: Ground truth tags of shape (batch, seq_len)

        Returns:
            Scalar loss
        """
        emissions = self.forward(x)
        return self.crf(emissions, tags)

    def decode(self, x: Tensor) -> Tensor:
        """Decode best tag sequence using Viterbi.

        Args:
            x: Input of shape (batch, seq_len, input_size)

        Returns:
            Best tags of shape (batch, seq_len)
        """
        emissions = self.forward(x)
        return self.crf.decode(emissions)

    def marginals(self, x: Tensor) -> Tensor:
        """Compute marginal probabilities P(y_t = k | x) via the CRF forward-backward.

        Args:
            x: Input of shape (batch, seq_len, input_size)

        Returns:
            Marginal probabilities of shape (batch, seq_len, num_tags)
        """
        emissions = self.forward(x)
        return self.crf.marginals(emissions)

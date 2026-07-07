"""Heart sound segmenter with a vendored xLSTM (mLSTM) emitter + CRF head (Experiment A).

Drop-in sibling of ``HeartSoundSegmenterCRF``: identical interface, CRF head, and decoders
(``loss`` / ``decode`` / ``marginals`` / ``decode_valid``) — only the recurrent emitter changes
from a 2-layer BiLSTM to the ``BidirectionalXLSTMEncoder`` (validated bit-for-bit against the
official ``xlstm`` package, see ``tests/test_xlstm_parity.py``). Because the encoder emits
``(B, T, 2 * hidden_size)``, the ``Linear(2H -> 4)`` emission head is unchanged.

Unlike the BiLSTM variant there are no registered ``h0``/``c0`` buffers — the mLSTM state is
initialized to zeros internally each forward — so this model has no batch-size-tied buffers and
accepts any batch size at eval time.
"""

import torch
from torch import nn

from hss.model.crf import CRF
from hss.model.xlstm import (
    BidirectionalXLSTMEncoder,
    CausalXLSTMEncoder,
    MultiRateXLSTMEncoder,
    PhaseBidirectionalXLSTMEncoder,
)
from hss.utils.sequence_validator import validate_and_correct_predictions


class HeartSoundSegmenterXLSTMCRF(nn.Module):
    """xLSTM emitter + CRF sequence model for 4-state heart-sound segmentation.

    Args:
        input_size: Size of input features at each time step.
        batch_size: Kept for interface parity with the BiLSTM models (unused; no h0/c0 buffers).
        hidden_size: Per-direction hidden width; encoder output is ``2 * hidden_size``.
        num_heads: mLSTM heads per layer (must divide ``hidden_size``).
        num_layers: Number of stacked bidirectional mLSTM layers.
        dropout: Dropout applied to the encoder output before the emission head.
        device: Device to place the model on.
        dtype: Data type for model parameters.
    """

    def __init__(
        self,
        *,
        input_size: int,
        batch_size: int = 1,
        hidden_size: int = 240,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        phase: bool = False,
        multirate: bool = False,
        rates: tuple[int, ...] = (1, 4, 16),
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.device = device if device is not None else torch.device("cpu")
        self.batch_size = batch_size
        self.num_tags = 4  # S1, Systole, S2, Diastole
        self.bidirectional = bidirectional
        self.phase = phase
        self.multirate = multirate

        if multirate:
            # Multi-rate temporal pyramid (Experiment D): fine/meso/coarse timescales, concatenated.
            self.encoder = MultiRateXLSTMEncoder(input_size, hidden_size, num_heads, num_layers, rates)
            enc_out = hidden_size * 2 * len(rates)
        elif phase:
            # Phase-clock mLSTM (Experiment C): cardiac-phase inductive bias inside the recurrence.
            self.encoder = PhaseBidirectionalXLSTMEncoder(input_size, hidden_size, num_heads, num_layers)
            enc_out = hidden_size * 2
        elif bidirectional:
            self.encoder = BidirectionalXLSTMEncoder(input_size, hidden_size, num_heads, num_layers)
            enc_out = hidden_size * 2
        else:
            # Causal (unidirectional) — streamable in real time (Experiment B); half the encoder width.
            self.encoder = CausalXLSTMEncoder(input_size, hidden_size, num_heads, num_layers)
            enc_out = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features=enc_out, out_features=self.num_tags, bias=True)

        self.crf = CRF(num_tags=self.num_tags)
        self._init_crf_transitions()

        self.to(device=self.device, dtype=dtype)

    def _init_crf_transitions(self) -> None:
        """Initialize CRF transition matrix with the cardiac-cycle prior.

        Valid transitions: 0->0, 0->1, 1->1, 1->2, 2->2, 2->3, 3->3, 3->0.
        Mirrors ``HeartSoundSegmenterCRF`` so the two models are comparable.
        """
        with torch.no_grad():
            self.crf.transitions.fill_(-1.0)
            self.crf.transitions[0, 0] = 1.0  # S1 -> S1
            self.crf.transitions[1, 1] = 1.0  # Systole -> Systole
            self.crf.transitions[2, 2] = 1.0  # S2 -> S2
            self.crf.transitions[3, 3] = 1.0  # Diastole -> Diastole
            self.crf.transitions[0, 1] = 1.0  # S1 -> Systole
            self.crf.transitions[1, 2] = 1.0  # Systole -> S2
            self.crf.transitions[2, 3] = 1.0  # S2 -> Diastole
            self.crf.transitions[3, 0] = 1.0  # Diastole -> S1

            self.crf.start_transitions.fill_(-1.0)
            self.crf.start_transitions[0] = 1.0  # S1
            self.crf.start_transitions[3] = 0.5  # Diastole

            self.crf.end_transitions.fill_(0.0)

    def _get_emissions(self, x: torch.Tensor) -> torch.Tensor:
        """Compute emission scores from input.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            Emission scores of shape (batch_size, sequence_length, num_tags).
        """
        output = self.encoder(x)
        output = self.dropout(output)
        return self.linear(output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning emission scores (batch_size, sequence_length, num_tags)."""
        return self._get_emissions(x)

    def loss(self, x: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood loss (scalar) for input ``x`` and target ``tags``."""
        emissions = self._get_emissions(x)
        return self.crf(emissions, tags)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Viterbi-decode the best tag sequence (batch_size, sequence_length)."""
        emissions = self._get_emissions(x)
        return self.crf.decode(emissions)

    def marginals(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-backward marginals P(y_t = k | x), shape (batch_size, sequence_length, num_tags)."""
        emissions = self._get_emissions(x)
        return self.crf.marginals(emissions)

    def decode_valid(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a guaranteed-valid cardiac cycle via constrained-posterior decoding.

        Constrained frame-level Viterbi (self-loops plus S1->Systole->S2->Diastole->S1 only) over
        the forward-backward posterior marginals — the same decoder the other models use, so all
        emitters are compared under one decode method.

        Returns:
            Best valid tag sequences (batch_size, sequence_length), labels 0-3.
        """
        marginals = self.marginals(x)
        log_posterior = torch.log(marginals.clamp_min(1e-9))
        corrected = torch.as_tensor(validate_and_correct_predictions(log_posterior), device=x.device)
        return corrected - 1

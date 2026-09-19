"""Encoder-decoder LSTM models and expert components."""

from __future__ import annotations

import torch
from torch import nn


class GlobalLSTM(nn.Module):
    """Global encoder-decoder LSTM for runoff prediction.

    Both the encoder and decoder receive the complete timestep input used in
    the model: dynamic meteorological inputs concatenated with
    static basin attributes. The final encoder hidden and cell states are
    transformed to initialize the decoder.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.hidden_transfer = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.cell_transfer = nn.Linear(hidden_size, hidden_size)
        self.regression_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        encoder_inputs: torch.Tensor,
        decoder_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Return runoff predictions with shape ``(batch, forecast_length)``."""

        _, (h_n, c_n) = self.encoder(encoder_inputs)

        init_h = self.hidden_transfer(h_n[-1])
        init_c = self.cell_transfer(c_n[-1])

        decoder_output, _ = self.decoder(
            decoder_inputs,
            (init_h.unsqueeze(0), init_c.unsqueeze(0)),
        )
        predictions = self.regression_head(decoder_output).squeeze(-1)
        return predictions


class LSTMExpert(nn.Module):
    """Encoder-decoder LSTM expert conditioned on static basin attributes.

    The MoE static encoder produces ``static_embedding``. Two independent
    linear projections add this embedding to the transformed encoder hidden
    and cell states before the decoder is run. Dropout is applied to those two
    initialized states before decoding.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float,
        static_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.decoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.hidden_transfer = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.cell_transfer = nn.Linear(hidden_size, hidden_size)
        self.static_to_h = nn.Linear(static_dim, hidden_size)
        self.static_to_c = nn.Linear(static_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.regression_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        encoder_inputs: torch.Tensor,
        decoder_inputs: torch.Tensor,
        static_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Return one expert's runoff prediction for each forecast timestep."""

        _, (h_n, c_n) = self.encoder(encoder_inputs)
        init_h = self.hidden_transfer(h_n[-1]) + self.static_to_h(static_embedding)
        init_c = self.cell_transfer(c_n[-1]) + self.static_to_c(static_embedding)
        init_h = self.dropout(init_h)
        init_c = self.dropout(init_c)
        decoder_output, _ = self.decoder(
            decoder_inputs,
            (init_h.unsqueeze(0), init_c.unsqueeze(0)),
        )
        predictions = self.regression_head(decoder_output).squeeze(-1)
        return predictions

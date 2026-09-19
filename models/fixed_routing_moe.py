"""Fixed-routing MoE model."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.lstm import LSTMExpert
from models.moe import StaticAttributeEncoder


class FixedRoutingMoE(nn.Module):
    """Route each sample to the LSTM expert fixed by its basin cluster."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float,
        static_input_dim: int,
        static_embed_dim: int,
        num_experts: int,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("At least one expert is required.")

        self.num_experts = num_experts
        self.static_encoder = StaticAttributeEncoder(static_input_dim, static_embed_dim)
        self.specialized_experts = nn.ModuleList(
            [
                LSTMExpert(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    static_dim=static_embed_dim,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(
        self,
        encoder_inputs: torch.Tensor,
        decoder_inputs: torch.Tensor,
        static_features: torch.Tensor,
        routing_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Return predictions using one pre-assigned expert per sample.

        ``routing_labels`` contains zero-based expert indices. Static basin
        attributes initialize the selected LSTM expert before decoding.
        """

        static_embedding = self.static_encoder(static_features)

        labels = routing_labels.to(device=encoder_inputs.device, dtype=torch.long)
        if labels.ndim != 1:
            labels = labels.view(-1)
        if labels.numel() != encoder_inputs.size(0):
            raise ValueError("routing_labels must contain one label per sample.")
        if torch.any((labels < 0) | (labels >= self.num_experts)):
            raise ValueError("routing_labels must be in [0, num_experts).")

        routing_probabilities = F.one_hot(labels, num_classes=self.num_experts).to(
            dtype=encoder_inputs.dtype
        )
        predictions = encoder_inputs.new_zeros(
            (encoder_inputs.size(0), decoder_inputs.size(1))
        )
        for expert_index, expert in enumerate(self.specialized_experts):
            sample_mask = labels == expert_index
            if sample_mask.any():
                expert_prediction = expert(
                    encoder_inputs[sample_mask],
                    decoder_inputs[sample_mask],
                    static_embedding[sample_mask],
                )
                predictions[sample_mask] = expert_prediction.to(predictions.dtype)

        load_balance_loss = predictions.sum() * 0.0
        routing = {
            "routing_probabilities": routing_probabilities,
            "topk_indices": labels.unsqueeze(-1).detach(),
            "topk_weights": torch.ones(
                (encoder_inputs.size(0), 1),
                dtype=encoder_inputs.dtype,
                device=encoder_inputs.device,
            ).detach(),
            "moe_prediction": predictions,
        }
        return predictions, load_balance_loss, routing

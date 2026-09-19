"""Learned-routing LSTM mixture-of-experts model with Top-2 aggregation."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from models.lstm import LSTMExpert


class StaticAttributeEncoder(nn.Module):
    """Encode static basin attributes as a fixed-size basin representation."""

    def __init__(self, input_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the static representation for each basin in the batch."""

        return self.net(x)


class MoE(nn.Module):
    """Top-2 learned routing over encoder-decoder LSTM experts.

    Static basin attributes determine dense routing probabilities. The two
    highest-probability experts retain nonzero, renormalized weights for runoff
    prediction. Dense probabilities are also returned for the load-balancing
    objective used during training.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float,
        static_input_dim: int,
        static_embed_dim: int,
        gating_hidden_dim: int,
        num_experts: int,
        top_k: int,
        temperature: float,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("At least one expert is required.")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must satisfy 1 <= top_k <= num_experts")

        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = max(temperature, 1e-3)
        self.eps = 1e-8

        # Keep module names stable for checkpoint loading.
        self.static_encoder = StaticAttributeEncoder(static_input_dim, static_embed_dim)
        self.gating_hidden = nn.Sequential(
            nn.Linear(static_embed_dim, gating_hidden_dim),
            nn.ReLU(),
        )
        self.specialized_head = nn.Linear(gating_hidden_dim, num_experts)
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
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Return prediction, load-balancing loss, and routing information."""

        static_embed = self.static_encoder(static_features)
        gating_hidden = self.gating_hidden(static_embed)
        routing_logits = self.specialized_head(gating_hidden) / self.temperature
        routing_probabilities = torch.softmax(routing_logits, dim=-1)

        if self.top_k == self.num_experts:
            sparse_routing_weights = routing_probabilities
            topk_indices = (
                torch.arange(self.num_experts, device=encoder_inputs.device)
                .unsqueeze(0)
                .expand_as(routing_probabilities)
            )
            topk_weights = sparse_routing_weights
        else:
            _, topk_indices = torch.topk(
                routing_probabilities, k=self.top_k, dim=-1
            )
            mask = torch.zeros_like(routing_probabilities).scatter(
                -1, topk_indices, 1.0
            )
            masked_routing_logits = torch.where(
                mask > 0,
                torch.log(routing_probabilities + self.eps),
                torch.full_like(routing_probabilities, -1e9),
            )
            sparse_routing_weights = torch.softmax(masked_routing_logits, dim=-1)
            topk_weights = torch.gather(
                sparse_routing_weights, dim=-1, index=topk_indices
            )

        # Aggregate expert predictions using Top-2 routing weights.
        expert_outputs = []
        for expert in self.specialized_experts:
            predictions = expert(encoder_inputs, decoder_inputs, static_embed)
            expert_outputs.append(predictions.unsqueeze(1))
        expert_outputs = torch.cat(expert_outputs, dim=1)
        moe_prediction = torch.sum(
            sparse_routing_weights.unsqueeze(-1) * expert_outputs, dim=1
        )

        load_balance_loss = self._load_balance_loss(routing_probabilities)
        routing_info = {
            "routing_probabilities": routing_probabilities,
            "routing_logits": routing_logits,
            "topk_indices": topk_indices.detach(),
            "topk_weights": topk_weights.detach(),
            "moe_prediction": moe_prediction,
        }
        return moe_prediction, load_balance_loss, routing_info

    def _load_balance_loss(self, probs: torch.Tensor) -> torch.Tensor:
        mean_usage = probs.mean(dim=0)
        uniform = torch.full_like(mean_usage, 1.0 / self.num_experts)
        return torch.sum(mean_usage * torch.log((mean_usage + self.eps) / uniform))

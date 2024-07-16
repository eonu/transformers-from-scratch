from __future__ import annotations

import torch
from torch import nn
from lightning import LightningModule

from transformer.params import TransformerParams, TransformerBlockParams
from transformer.modules.attention import MultiHeadSelfAttention

__all__ = ["Transformer", "TransformerBlock"]


class Transformer(LightningModule):
    def __init__(
        self: Transformer,
        params: TransformerParams | None = None,
    ) -> Transformer:
        self.params: TransformerParams = params or TransformerParams()
        self.model: nn.Sequential = nn.Sequential(
            TransformerBlock(self.params.block_params)
            for _ in range(self.params.num_blocks)
        )

    def forward(self: Transformer, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.model(x)
        # shape: [batch_size, model_dim]


class TransformerBlock(LightningModule):
    def __init__(
        self: TransformerBlock, 
        params: TransformerBlockParams | None = None,
    ) -> TransformerBlock:
        self.params: TransformerBlockParams = params or TransformerBlockParams()
        self.model: nn.ModuleDict = nn.ModuleDict({
            "attn": MultiHeadSelfAttention(self.params.multi_head_params),
            "norm_attn": nn.LayerNorm(self.params.model_dim),
            "ffnn": nn.Sequential(
                nn.Linear(self.params.model_dim, self.params.feed_forward_dim),
                nn.ReLU(),
                nn.Linear(self.params.feed_forward_dim, self.params.model_dim),
            ),
            "norm_ffnn": nn.LayerNorm(self.params.model_dim)
        })

    def forward(self: TransformerBlock, x: torch.FloatTensor) -> torch.FloatTensor:
        o = self.model["norm_attn"](x + self.model["attn"](q=x, k=x, v=x))
        # shape: [batch_size, model_dim]
        h = self.model["norm_ffnn"](o + self.model["ffn"](o))
        return h
        # shape: [batch_size, model_dim]

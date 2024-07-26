from __future__ import annotations

import torch
from torch import nn
from lightning import LightningModule

from transformer.params import TransformerParams, TransformerBlockParams
from transformer.modules.attention import MultiHeadSelfAttention

__all__ = ["Transformer", "TransformerBlock"]


class Transformer(LightningModule):
    """TODO: docstring"""

    def __init__(
        self: Transformer,
        params: TransformerParams,
    ) -> Transformer:
        super().__init__()
        self.params: TransformerParams = params
        self.model = nn.Sequential(
            *[
                TransformerBlock(self.params.block_params)
                for _ in range(self.params.num_blocks)
            ]
        )

    def forward(
        self: Transformer, x: torch.FloatTensor
    ) -> torch.FloatTensor:
        return self.model(x)
        # shape: [batch_size, context_length, model_dim]


class TransformerBlock(LightningModule):
    """TODO: docstring"""

    def __init__(
        self: TransformerBlock,
        params: TransformerBlockParams = TransformerBlockParams(),
    ) -> TransformerBlock:
        super().__init__()
        self.params: TransformerBlockParams = params
        self.model: nn.ModuleDict = nn.ModuleDict(
            {
                "attn": MultiHeadSelfAttention(
                    self.params.multi_head_params, mask=False
                ),
                "attn_dropout": nn.Dropout(0.1),
                "norm_attn": nn.LayerNorm(self.params.model_dim),
                "ffnn": nn.Sequential(
                    nn.Linear(self.params.model_dim, self.params.feed_forward_dim),
                    nn.ReLU(),
                    nn.Linear(self.params.feed_forward_dim, self.params.model_dim),
                    nn.Dropout(0.1),
                ),
                "ffnn_dropout": nn.Dropout(0.1),
                "norm_ffnn": nn.LayerNorm(self.params.model_dim),
            }
        )

    def forward(self: TransformerBlock, x: torch.FloatTensor) -> torch.FloatTensor:
        # calculate attention vectors, add residual and normalize
        attn = self.model["attn_dropout"](self.model["attn"](q=x, k=x, v=x))
        attn = self.model["norm_attn"](x + attn)
        # shape: [batch_size, context_length, model_dim]

        # pass attention vectors through FFNN, add residual and normalize
        hidden = self.model["ffnn_dropout"](self.model["ffnn"](attn))
        return self.model["norm_ffnn"](attn + hidden)
        # shape: [batch_size, context_length, model_dim]

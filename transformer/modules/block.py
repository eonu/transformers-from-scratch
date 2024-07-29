from __future__ import annotations

import typing as t

import torch
from torch import nn
from lightning import LightningModule

from transformer.params import TransformerBlockParams
from transformer.modules.attention import MultiHeadSelfAttention

__all__ = ["TransformerBlock"]


class TransformerBlock(LightningModule):
    """TODO: docstring
    NOTE: Only used for encoder-only/decoder-only, encoder-decoder has its own block"""

    def __init__(self: t.Self, params: TransformerBlockParams) -> None:
        super().__init__()
        self.params: TransformerBlockParams = params
        self.model: nn.ModuleDict = nn.ModuleDict(
            {
                "attn": nn.ModuleDict(
                    {
                        "attn": MultiHeadSelfAttention(self.params.multi_head_params),
                        "dropout": nn.Dropout(0.1),
                        "norm": nn.LayerNorm(self.params.model_dim),
                    }
                ),
                "ffnn": nn.ModuleDict(
                    {
                        "fc": nn.Sequential(
                            nn.Linear(
                                self.params.model_dim, self.params.feed_forward_dim
                            ),
                            nn.ReLU(),
                            nn.Linear(
                                self.params.feed_forward_dim, self.params.model_dim
                            ),
                        ),
                        "dropout": nn.Dropout(0.1),
                        "norm": nn.LayerNorm(self.params.model_dim),
                    }
                ),
            }
        )

    @property
    def attn(self: t.Self) -> nn.ModuleDict:
        return self.model["attn"]

    @property
    def ffnn(self: t.Self) -> nn.ModuleDict:
        return self.model["ffnn"]

    def forward(
        self: t.Self, x: torch.FloatTensor, masks: torch.LongTensor
    ) -> torch.FloatTensor:
        # calculate attention vectors, add residual and normalize
        attn = self.attn["dropout"](self.attn["attn"](q=x, k=x, v=x, masks=masks))
        attn = self.attn["norm"](x + attn)
        # shape: [batch_size, context_length, model_dim]

        # pass attention vectors through FFNN, add residual and normalize
        hidden = self.ffnn["dropout"](self.ffnn["fc"](attn))
        return self.ffnn["norm"](attn + hidden)
        # shape: [batch_size, context_length, model_dim]

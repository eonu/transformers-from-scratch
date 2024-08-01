from __future__ import annotations

import typing as t

import torch
from torch import nn
from lightning import LightningModule

from transformer.params import TransformerParams
from transformer.modules.block import TransformerBlock

__all__ = ["BaseTransformer"]


class BaseTransformer(LightningModule):
    def __init__(self: t.Self, params: TransformerParams) -> None:
        super().__init__()
        self.params: TransformerParams = params
        self.model = nn.ModuleList(
            [
                TransformerBlock(self.params.block_params)
                for _ in range(self.params.num_blocks)
            ]
        )

    def forward(
        self: t.Self, x: torch.FloatTensor, masks: torch.LongTensor
    ) -> torch.FloatTensor:
        for block in self.model:
            x = block(x=x, masks=masks)
        return x
        # shape: [batch_size, context_length, model_dim]

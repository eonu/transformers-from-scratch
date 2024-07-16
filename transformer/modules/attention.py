"""TODO: docstring"""

from __future__ import annotations

import math

import torch
from torch import nn
from lightning import LightningModule

from transformer.params import SelfAttentionParams, MultiHeadSelfAttentionParams

__all__ = ["MultiHeadSelfAttention", "SelfAttention"]


class MultiHeadSelfAttention(LightningModule):
    """TODO: docstring"""

    def __init__(
        self: MultiHeadSelfAttention,
        params: MultiHeadSelfAttentionParams = MultiHeadSelfAttentionParams(),
        *,
        mask: bool = True,
    ) -> MultiHeadSelfAttention:
        super().__init__()
        self.params: MultiHeadSelfAttentionParams = params
        self.mask: bool = mask
        self.model: nn.ModuleDict = nn.ModuleDict(
            {
                "heads": nn.ModuleList(
                    SelfAttention(self.params.attention_params, mask=mask)
                    for _ in range(self.params.num_heads)
                ),
                "proj": nn.Linear(self.params.model_dim, self.params.model_dim),
            }
        )

    def forward(
        self: MultiHeadSelfAttention,
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        v: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # concatenate attention head outputs
        heads = torch.hstack([head(q, k, v) for head in self.model["heads"]])
        # shape: [batch_size, value_dim * num_heads]

        # project onto output matrix
        return self.model["proj"](heads)
        # shape: [batch_size, model_dim]


class SelfAttention(LightningModule):
    """TODO: docstring"""

    def __init__(
        self: SelfAttention,
        params: SelfAttentionParams = SelfAttentionParams(),
        *,
        mask: bool = True,
    ) -> SelfAttention:
        super().__init__()
        self.params: SelfAttentionParams = params
        self.mask: bool = mask
        self.model: nn.ModuleDict = nn.ModuleDict(
            {
                "query_proj": nn.Linear(
                    self.params.model_dim, self.params.key_dim, bias=False
                ),
                "key_proj": nn.Linear(
                    self.params.model_dim, self.params.key_dim, bias=False
                ),
                "value_proj": nn.Linear(
                    self.params.model_dim, self.params.value_dim, bias=False
                ),
            }
        )

    def forward(
        self: MultiHeadSelfAttention,
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        v: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # project inputs onto weight matrices
        q = self.model["query_proj"](q)  # shape: [batch_size, key_dim]
        k = self.model["key_proj"](k)  # shape: [batch_size, key_dim]
        v = self.model["value_proj"](v)  # shape: [batch_size, value_dim]

        # calculate scores, i.e. scaled dot products
        scores = q @ k.T / math.sqrt(self.params.key_dim)
        # shape: [batch_size, batch_size]

        if self.mask:
            # mask upper diagonal before softmax to prevent looking into future
            scores += torch.triu(-torch.inf * torch.ones_like(scores), diagonal=1)
            # shape: [batch_size, batch_size]

        return nn.functional.softmax(scores, dim=0) @ v
        # shape: [batch_size, value_dim]

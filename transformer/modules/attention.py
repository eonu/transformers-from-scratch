"""TODO: docstring"""

from __future__ import annotations

import math
import typing as t

import torch
from torch import nn
from lightning import LightningModule

from transformer.params import SelfAttentionParams, MultiHeadSelfAttentionParams

__all__ = ["MultiHeadSelfAttention", "SelfAttention"]

EPS = torch.finfo().eps


class MultiHeadSelfAttention(LightningModule):
    """TODO: docstring"""

    def __init__(self: t.Self, params: MultiHeadSelfAttentionParams) -> None:
        super().__init__()
        self.params: MultiHeadSelfAttentionParams = params
        self.model: nn.ModuleDict = nn.ModuleDict(
            {
                "heads": nn.ModuleList(
                    SelfAttention(self.params.attention_params)
                    for _ in range(self.params.num_heads)
                ),
                "proj": nn.Linear(self.params.model_dim, self.params.model_dim),
            }
        )

    def forward(
        self: t.Self,
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        v: torch.FloatTensor,
        masks: torch.LongTensor,
    ) -> torch.FloatTensor:
        # concatenate attention head outputs
        heads = torch.cat(
            [head(q, k, v, masks=masks) for head in self.model["heads"]], dim=-1
        )
        # shape: [batch_size, context_length, value_dim * num_heads (= model_dim)]

        # project onto output matrix
        return self.model["proj"](heads)
        # shape: [batch_size, context_length, model_dim]


class SelfAttention(LightningModule):
    """TODO: docstring"""

    def __init__(self: t.Self, params: SelfAttentionParams) -> None:
        super().__init__()
        self.params: SelfAttentionParams = params
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
        self: t.Self,
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        v: torch.FloatTensor,
        masks: torch.LongTensor,
    ) -> torch.FloatTensor:
        # project inputs onto weight matrices
        q = self.model["query_proj"](q)
        k = self.model["key_proj"](k)
        # shapes: [batch_size, context_length, key_dim]
        v = self.model["value_proj"](v)
        # shape: [batch_size, context_length, value_dim]

        # calculate scores, i.e. scaled dot products
        scores = q @ k.mT / math.sqrt(self.params.key_dim)
        # shape: [batch_size, context_length, context_length]

        # tokenizer attention mask to ignore padding (a.k.a. key padding mask)
        attn_mask = 1 - masks.unsqueeze(1) * masks.unsqueeze(-1)
        # shape: [batch_size, context_length, context_length]

        # upper-diagonal lookahead mask before softmax to prevent looking into future
        if self.params.mask:
            attn_mask |= torch.triu(torch.ones_like(scores, dtype=int), diagonal=1)
            # shape: [batch_size, context_length, context_length]

        # apply mask(s)
        scores.masked_fill_(attn_mask.bool(), -EPS)

        # compute scores
        return nn.functional.softmax(scores, dim=-1) @ v
        # shape: [batch_size, context_length, value_dim]

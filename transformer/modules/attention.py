"""TODO: docstring"""

from __future__ import annotations

import torch
import pydantic as pyd
from torch import nn

from transformer.params import SelfAttentionParams, MultiHeadSelfAttentionParams

__all__ = ["MultiHeadSelfAttention", "SelfAttention"]

class MultiHeadSelfAttention:
    """TODO: docstring"""

    def __init__(
        self: MultiHeadSelfAttention,
        params: MultiHeadSelfAttentionParams | None = None,
        *,
        mask: bool = True,
    ) -> MultiHeadSelfAttention:
        self.params: MultiHeadSelfAttentionParams = params or MultiHeadSelfAttentionParams()
        self.mask: bool = mask
        self.model: nn.ModuleDict = nn.ModuleDict({
            "heads": nn.ModuleList(
                SelfAttention(self.params.attention_params, mask=mask)
                for _ in range(self.params.num_heads)
            ),
            "proj": nn.Linear(
                self.params.value_dim * self.params.num_heads, 
                self.params.model_dim,
            ),
        })

    def forward(
        self: MultiHeadSelfAttention,
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        v: torch.FloatTensor,
    ) -> torch.FloatTensor:
        heads = torch.hstack([head(q, k, v) for head in self.model["heads"]])
        # shape: [batch_size, value_dim * num_heads]
        return self.model["proj"](heads)
        # shape: [batch_size, model_dim]


class SelfAttention:
    """TODO: docstring"""

    def __init__(
        self: SelfAttention, 
        params: SelfAttentionParams | None = None,
        *,
        mask: bool = True,
    ) -> SelfAttention:
        self.params: SelfAttentionParams = params or SelfAttentionParams()
        self.mask: bool = mask
        # TODO @eonu


    def forward(
        self: MultiHeadSelfAttention,
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        v: torch.FloatTensor,
    ) -> torch.FloatTensor:
        pass
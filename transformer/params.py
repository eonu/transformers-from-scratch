"""TODO: docstring"""

from __future__ import annotations

import pydantic as pyd

__all__ = [
    "TransformerParams",
    "TransformerBlockParams",
    "SelfAttentionParams",
    "MultiHeadSelfAttentionParams",
]


class TransformerParams(pyd.BaseModel, frozen=True, protected_namespaces=()):
    """TODO: docstring"""

    # transformer level params
    num_blocks: pyd.PositiveInt = 6
    model_dim: pyd.PositiveInt = 512
    # transformer block level params
    feed_forward_dim: pyd.PositiveInt = 2_048
    # multi-head self-attention level params
    num_heads: pyd.PositiveInt = 8

    @property
    def block_params(
        self: TransformerBlockParams,
    ) -> TransformerBlockParams:
        return TransformerBlockParams(
            model_dim=self.model_dim,
            feed_forward_dim=self.feed_forward_dim,
            num_heads=self.num_heads,
        )


class TransformerBlockParams(pyd.BaseModel, frozen=True, protected_namespaces=()):
    """TODO: docstring"""

    # transformer level params
    model_dim: pyd.PositiveInt = 512
    # transformer block level params
    feed_forward_dim: pyd.PositiveInt = 2_048
    # multi-head self-attention level params
    num_heads: pyd.PositiveInt = 8

    @property
    def multi_head_params(
        self: MultiHeadSelfAttentionParams,
    ) -> MultiHeadSelfAttentionParams:
        return MultiHeadSelfAttentionParams(
            model_dim=self.model_dim, num_heads=self.num_heads
        )


class MultiHeadSelfAttentionParams(pyd.BaseModel, frozen=True, protected_namespaces=()):
    """TODO: docstring"""

    # transformer level params
    model_dim: pyd.PositiveInt = 512
    # multi-head self-attention level params
    num_heads: pyd.PositiveInt = 8

    @property
    def attention_params(
        self: SelfAttentionParams,
    ) -> SelfAttentionParams:
        return SelfAttentionParams(model_dim=self.model_dim, num_heads=self.num_heads)


class SelfAttentionParams(pyd.BaseModel, frozen=True, protected_namespaces=()):
    """TODO: docstring"""

    # transformer level params
    model_dim: pyd.PositiveInt = 512
    # multi-head self-attention level params
    num_heads: pyd.PositiveInt = 8

    @property
    def key_dim(self: SelfAttentionParams) -> int:
        return self.model_dim // self.num_heads

    @property
    def value_dim(self: SelfAttentionParams) -> int:
        return self.model_dim // self.num_heads

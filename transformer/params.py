"""TODO: docstring"""

from __future__ import annotations

from pydantic import pyd

__all__ = ["TransformerParams", "SelfAttentionParams", "MultiHeadSelfAttentionParams"]

class TransformerParams(pyd.BaseModel, frozen=True):
    """TODO: docstring"""
    # transformer level params
    num_blocks: pyd.PositiveInt = 6
    model_dim: pyd.PositiveInt = 512
    # transformer block level params
    feed_forward_dim: pyd.PositiveInt = 2_048
    # multi-head self-attention level params
    num_heads: pyd.PositiveInt = 8
    # self-attention level params
    key_dim: pyd.PositiveInt = 64
    value_dim: pyd.PositiveInt = 64

    @property
    def block_params(
        self: TransformerBlockParams,
    ) -> TransformerBlockParams:
        return TransformerBlockParams(
            model_dim=self.model_dim,
            feed_forward_dim=self.feed_forward_dim,
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
        )

class TransformerBlockParams(pyd.BaseModel, frozen=True):
    """TODO: docstring"""
    # transformer level params
    model_dim: pyd.PositiveInt = 512
    # transformer block level params 
    feed_forward_dim: pyd.PositiveInt = 2_048
    # multi-head self-attention level params
    num_heads: pyd.PositiveInt = 8
    # self-attention level params
    key_dim: pyd.PositiveInt = 64
    value_dim: pyd.PositiveInt = 64

    @property
    def multi_head_params(
        self: MultiHeadSelfAttentionParams,
    ) -> MultiHeadSelfAttentionParams:
        return MultiHeadSelfAttentionParams(
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
        )
    
class MultiHeadSelfAttentionParams(pyd.BaseModel, frozen=True):
    """TODO: docstring"""
    # transformer level params
    model_dim: pyd.PositiveInt = 512
    # multi-head self-attention level params
    num_heads: pyd.PositiveInt = 8
    # self-attention level params
    key_dim: pyd.PositiveInt = 64
    value_dim: pyd.PositiveInt = 64

    @property
    def attention_params(
        self: SelfAttentionParams,
    ) -> SelfAttentionParams:
        return SelfAttentionParams(
            model_dim=self.model_dim,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
        )


class SelfAttentionParams(pyd.BaseModel, frozen=True):
    """TODO: docstring"""
    # transformer level params
    model_dim: pyd.PositiveInt = 512
    # self-attention level params
    key_dim: pyd.PositiveInt = 64
    value_dim: pyd.PositiveInt = 64

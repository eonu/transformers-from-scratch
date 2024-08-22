from __future__ import annotations

import typing as t

import pydantic as pyd

__all__ = [
    "TransformerParams",
    "TransformerBlockParams",
    "SelfAttentionParams",
    "MultiHeadSelfAttentionParams",
]


class MaskableParams(pyd.BaseModel, frozen=True, protected_namespaces=()):
    # self-attention level params
    mask: bool | None = None

    @property
    def masked(self: t.Self) -> MaskableParams:
        return self.__class__(**self.model_dump(exclude="mask"), mask=True)

    @property
    def unmasked(self: t.Self) -> MaskableParams:
        return self.__class__(**self.model_dump(exclude="mask"), mask=False)


class TransformerParams(MaskableParams):
    # transformer level params
    context_length: pyd.PositiveInt
    num_blocks: pyd.PositiveInt = 6
    model_dim: pyd.PositiveInt = 512
    # transformer block level params
    feed_forward_dim: pyd.PositiveInt = 2_048
    # multi-head self-attention level params
    num_heads: pyd.PositiveInt = 8

    @property
    def block_params(self: t.Self) -> TransformerBlockParams:
        return TransformerBlockParams(
            model_dim=self.model_dim,
            feed_forward_dim=self.feed_forward_dim,
            num_heads=self.num_heads,
            mask=self.mask,
        )


class TransformerBlockParams(MaskableParams):
    # transformer level params
    model_dim: pyd.PositiveInt
    # transformer block level params
    feed_forward_dim: pyd.PositiveInt
    # multi-head self-attention level params
    num_heads: pyd.PositiveInt

    @property
    def multi_head_params(self: t.Self) -> MultiHeadSelfAttentionParams:
        return MultiHeadSelfAttentionParams(
            model_dim=self.model_dim, num_heads=self.num_heads, mask=self.mask
        )


class MultiHeadSelfAttentionParams(MaskableParams):
    # transformer level params
    model_dim: pyd.PositiveInt
    # multi-head self-attention level params
    num_heads: pyd.PositiveInt

    @property
    def attention_params(self: t.Self) -> SelfAttentionParams:
        return SelfAttentionParams(
            model_dim=self.model_dim, num_heads=self.num_heads, mask=self.mask
        )


class SelfAttentionParams(MaskableParams):
    # transformer level params
    model_dim: pyd.PositiveInt
    # multi-head self-attention level params
    num_heads: pyd.PositiveInt

    @property
    def key_dim(self: t.Self) -> int:
        return self.model_dim // self.num_heads

    @property
    def value_dim(self: t.Self) -> int:
        return self.model_dim // self.num_heads

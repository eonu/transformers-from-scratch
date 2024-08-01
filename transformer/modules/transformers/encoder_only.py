from __future__ import annotations

import typing as t

from transformer.params import TransformerParams
from transformer.modules.transformers.base import BaseTransformer


__all__ = ["EncoderTransformer"]


class EncoderTransformer(BaseTransformer):
    def __init__(self: t.Self, params: TransformerParams) -> None:
        super().__init__(params=params.unmasked)

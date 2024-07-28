from __future__ import annotations

from transformer.params import TransformerParams
from transformer.modules.transformers.base import BaseTransformer


__all__ = ["EncoderTransformer"]


class EncoderTransformer(BaseTransformer):
    """TODO: docstring"""

    def __init__(
        self: EncoderTransformer, params: TransformerParams
    ) -> EncoderTransformer:
        super().__init__(params=params.unmasked)

from __future__ import annotations

from transformer.params import TransformerParams
from transformer.modules.transformers.base import BaseTransformer

__all__ = ["DecoderTransformer"]


class DecoderTransformer(BaseTransformer):
    """TODO: docstring"""

    def __init__(
        self: DecoderTransformer, params: TransformerParams
    ) -> DecoderTransformer:
        super().__init__(params=params.masked)

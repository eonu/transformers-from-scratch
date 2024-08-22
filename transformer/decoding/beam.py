import typing as t

import torch
import numpy as np
import pydantic as pyd
from transformers import PreTrainedTokenizer

from transformer.decoding.base import BaseDecoder
from transformer.params import BeamSearchParams
from transformer.models import CausalLM, Seq2SeqLM

__all__ = ["BeamSearchDecoder"]


class BeamSearchDecoder(BaseDecoder):
    @pyd.validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self: t.Self,
        params: BeamSearchParams,
        model: CausalLM | Seq2SeqLM,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        super().__init__(params=params, model=model, random_state=random_state)

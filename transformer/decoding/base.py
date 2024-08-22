import abc
import typing as t
from operator import itemgetter

import torch
import numpy as np
from sklearn.utils import check_random_state
from transformers import PreTrainedTokenizer

from transformer.params import DecodingParams
from transformer.models import CausalLM, Seq2SeqLM

__all__ = ["BaseDecoder"]


class BaseDecoder(metaclass=abc.ABCMeta):
    def __init__(
        self: t.Self,
        params: DecodingParams,
        model: CausalLM | Seq2SeqLM,
        random_state: int | np.random.RandomState | None,
    ) -> None:
        self.params: DecodingParams = params
        self.model: CausalLM | Seq2SeqLM = model.eval()
        self.random_state: np.random.RandomState = check_random_state(random_state)

    @abc.abstractmethod
    def _generate(
        self: t.Self,
        context: str | None,
        /,
        *,
        tokenizer: PreTrainedTokenizer,
        forward: t.Callable[[torch.LongTensor, torch.LongTensor], torch.FloatTensor],
    ) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, int]:
        if self.model.params.context_length > self.params.max_length:
            raise ValueError(
                "Maximum sequence length must be longer than context length"
            )

        # encode input
        tokens = tokenizer(
            context or "",
            add_special_tokens=True,
            padding="max_length",
            max_length=self.model.params.context_length,
            return_tensors="pt",
        )
        output_ids, output_masks = itemgetter("input_ids", "attention_mask")(tokens)

        if output_masks.sum() >= self.model.params.context_length:
            raise ValueError("Provdided context must be shorter than context length")

        # update output IDs and mask to skip <eos> token
        eos_idx = output_masks.argmin() - 1
        output_ids[0, eos_idx] = tokenizer.pad_token_id
        output_masks[0, eos_idx] = 0

        # prepare output array
        output: torch.LongTensor = torch.full(
            size=(self.params.max_length,), fill_value=tokenizer.pad_token_id
        )
        output[:eos_idx] = output_ids[0, :eos_idx].clone()

        return output, output_ids, output_masks, eos_idx

    def generate(self: t.Self, context: str | None = None, /) -> str:
        if not isinstance(self.model, CausalLM):
            raise TypeError("Autoregressive generation requires a CausalLM")
        with torch.no_grad():
            return self._generate(
                context, tokenizer=self.model.tokenizer, forward=self.model.forward
            )

    def convert(self: t.Self, input_string: str, /, context: str | None = None) -> str:
        if not isinstance(self.model, Seq2SeqLM):
            raise TypeError("Sequence-to-sequence generation requires a Seq2SeqLM")

        # tokenize input string
        tokens: dict[str, torch.LongTensor] = self.model.input_tokenizer(
            input_string,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.model.params.context_length,
            return_tensors="pt",
        )
        input_ids, input_masks = itemgetter("input_ids", "attention_mask")(tokens)

        # curry forward function to only require output IDs and output masks
        with torch.no_grad():
            return self._generate(
                context,
                tokenizer=self.model.output_tokenizer,
                forward=lambda output_ids, output_masks: self.model.forward(
                    input_ids=input_ids,
                    output_ids=output_ids,
                    input_masks=input_masks,
                    output_masks=output_masks,
                ),
            )

import typing as t

import torch
import numpy as np
import pydantic as pyd
from transformers import PreTrainedTokenizer

from transformer.decoding.base import BaseDecoder
from transformer.params import GreedySearchParams
from transformer.models import CausalLM, Seq2SeqLM


__all__ = ["GreedySearchDecoder"]


class GreedySearchDecoder(BaseDecoder):
    @pyd.validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self: t.Self,
        params: GreedySearchParams,
        model: CausalLM | Seq2SeqLM,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        super().__init__(params=params, model=model, random_state=random_state)

    def _generate(
        self: t.Self,
        context: str | None,
        /,
        *,
        tokenizer: PreTrainedTokenizer,
        forward: t.Callable[[torch.LongTensor, torch.LongTensor], torch.FloatTensor],
    ) -> str:
        # tokenize context (if any)
        output, output_ids, output_masks, length = super()._generate(
            context, tokenizer=tokenizer, forward=forward
        )

        # valid ids to sample from
        valid_ids = torch.tensor(
            [
                token_id
                for token_id in range(len(tokenizer))
                if token_id
                not in (
                    tokenizer.bos_token_id,
                    tokenizer.pad_token_id,
                    tokenizer.unk_token_id,
                )
            ]
        )

        while (
            length < self.params.max_length
            and output[length - 1] != tokenizer.eos_token_id
        ):
            # make prediction
            pred = forward(output_ids, output_masks)

            # pick token with highest log-softmax score
            idx = -1 if length >= self.model.params.context_length else length - 1
            log_probs = pred[0, idx, valid_ids]
            next_token_id = valid_ids[log_probs.argmax()]

            # append to output
            output[length] = next_token_id

            # shift context window and masks (if necessary) and update
            if length >= self.model.params.context_length:
                output_ids[0, :-1] = output_ids[0, 1:].clone()
                output_ids[0, -1] = next_token_id
            else:
                output_masks[0, length] = 1
                output_ids[0, length] = next_token_id

            length += 1

        # decode IDs and untokenize back to string
        output_string = tokenizer.decode(
            output.tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # if didn't terminate, add ellipsis at end to indicate
        if output[length - 1] != tokenizer.eos_token_id:
            output_string = f"{output_string}..."

        return output_string

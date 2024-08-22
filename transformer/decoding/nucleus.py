import typing as t

import torch
import numpy as np
import pydantic as pyd
from transformers import PreTrainedTokenizer

from transformer.decoding.base import BaseDecoder
from transformer.params import NucleusSamplingParams
from transformer.models import CausalLM, Seq2SeqLM


__all__ = ["NucleusSamplingDecoder"]


class NucleusSamplingDecoder(BaseDecoder):
    @pyd.validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self: t.Self,
        params: NucleusSamplingParams,
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
        valid_ids = self._valid_ids(tokenizer)

        while (
            length < self.params.max_length
            and output[length - 1] != tokenizer.eos_token_id
        ):
            # make prediction
            pred = forward(output_ids, output_masks)

            # get log probabilities for next token
            idx = -1 if length >= self.model.params.context_length else length - 1
            log_probs = pred[0, idx, valid_ids]

            # sort log_probabilities
            sorted_log_probs, sort_idxs = log_probs.sort(descending=True)

            # find first index such that cumulative probability is greater than or equal to p
            prob_idx = -1
            prob_ind = sorted_log_probs.exp().cumsum(dim=0) > self.params.p
            if prob_ind.any():
                prob_idx = prob_ind.float().argmax() + 1

            # restrict the log probabilities
            selected_log_probs = sorted_log_probs[:prob_idx]
            selected_idxs = sort_idxs[:prob_idx]

            # sample from categorical distribution using the Gumbel-max trick
            gumbels = self.random_state.gumbel(size=len(selected_idxs))
            idx = np.argmax(selected_log_probs + gumbels)
            next_token_id = valid_ids[selected_idxs[idx]]

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

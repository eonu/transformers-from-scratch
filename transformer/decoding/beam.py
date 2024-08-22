import heapq
import typing as t

import torch
import numpy as np
import pydantic as pyd
from transformers import PreTrainedTokenizer

from transformer.decoding.base import BaseDecoder
from transformer.params import BeamSearchParams
from transformer.models import CausalLM, Seq2SeqLM

__all__ = ["BeamSearchDecoder"]


class Path(t.NamedTuple):
    log_prob: float
    window: torch.LongTensor
    mask: torch.LongTensor
    output: torch.LongTensor
    length: int
    terminated: bool

    def __lt__(self: t.Self, other: t.Any) -> bool:
        return self.log_prob < other.log_prob


class BeamSearchDecoder(BaseDecoder):
    @pyd.validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self: t.Self,
        params: BeamSearchParams,
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

        # make prediction
        pred = forward(output_ids, output_masks)

        # initialize heap with tokens with k highest log-softmax scores
        paths: list[Path] = []
        log_probs = pred[0, length - 1, valid_ids].topk(k=self.params.beam_width)
        for i in range(self.params.beam_width):
            log_prob: torch.FloatTensor = log_probs.values[i].item()
            token_id: torch.LongTensor = valid_ids[log_probs.indices][i]
            path_output: torch.LongTensor = output.clone()
            path_output[length] = token_id
            window: torch.LongTensor = output_ids.clone()
            window[0, length] = token_id
            mask: torch.LongTensor = output_masks.clone()
            mask[0, length] = 1
            heapq.heappush(
                paths, Path(log_prob, window, mask, path_output, length + 1, False)
            )

        while paths:
            # if most likely path is terminated, return it
            if paths[0].terminated:
                output = paths[0].output
                break

            new_paths: list[Path] = []
            while paths:
                # pick the current least likely path
                path = heapq.heappop(paths)

                # don't expand terminated paths
                if path.terminated:
                    if len(new_paths) < self.params.beam_width:
                        # push new path
                        heapq.heappush(new_paths, path)
                    else:
                        # replace least likely path if more probable
                        if path.log_prob >= new_paths[0].log_prob:
                            heapq.heapreplace(new_paths, path)
                    continue

                # make prediction
                pred = forward(path.window, path.mask)

                # get top-k most likely tokens
                idx = (
                    -1
                    if path.length >= self.model.params.context_length
                    else path.length - 1
                )
                log_probs = pred[0, idx, valid_ids].topk(k=self.params.beam_width)

                # expand path
                for i in range(self.params.beam_width):
                    next_log_prob: torch.FloatTensor = log_probs.values[i].item()

                    # don't expand if new log probability is lower than the current lowest
                    if (
                        len(new_paths) == self.params.beam_width
                        and next_log_prob < new_paths[0].log_prob
                    ):
                        continue

                    # copy output and set next token ID
                    next_token_id: torch.LongTensor = valid_ids[log_probs.indices][i]
                    path_output: torch.LongTensor = path.output.clone()
                    path_output[path.length] = next_token_id
                    # shift context window and masks (if necessary) and update
                    window: torch.LongTensor = path.window.clone()
                    mask: torch.LongTensor = path.mask.clone()
                    if path.length >= self.model.params.context_length:
                        window[0, :-1] = window[0, 1:].clone()
                        window[0, -1] = next_token_id
                    else:
                        mask[0, path.length] = 1
                        window[0, path.length] = next_token_id
                    # whether reached maximum length or <eos> token
                    terminated = (
                        next_token_id == tokenizer.eos_token_id
                        or path.length + 1 == self.params.max_length
                    )

                    # create new path
                    new_path = Path(
                        log_prob + next_log_prob,
                        window,
                        mask,
                        path_output,
                        path.length + 1,
                        terminated,
                    )

                    if len(new_paths) < self.params.beam_width:
                        # push new path
                        heapq.heappush(new_paths, new_path)
                    else:
                        # replace least likely path if more probable
                        # note: wouldn't have reached here if there was a more likely path
                        heapq.heapreplace(new_paths, new_path)

            # update current candidates
            paths = new_paths

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

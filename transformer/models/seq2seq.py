from __future__ import annotations

import typing as t
from operator import itemgetter

from transformer.modules.transformers.encoder_decoder import EncoderDecoderTransformer
from transformer.modules.embedding import InputEmbedding
from transformer.params import TransformerParams

from transformers import PreTrainedTokenizer

import torch
from torch import nn
from lightning import LightningModule

__all__ = ["Seq2SeqLM"]


class Seq2SeqLM(LightningModule):
    def __init__(
        self: t.Self,
        config: TransformerParams,
        input_tokenizer: PreTrainedTokenizer,
        output_tokenizer: PreTrainedTokenizer,
    ) -> None:
        super().__init__()
        self.config = config
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        self.model = nn.ModuleDict(
            {
                "input": nn.ModuleDict(
                    {
                        "emb": InputEmbedding(
                            len(self.input_tokenizer), config.model_dim
                        ),
                        "dropout": nn.Dropout(0.1),
                    }
                ),
                "output": nn.ModuleDict(
                    {
                        "emb": InputEmbedding(
                            len(self.input_tokenizer), config.model_dim
                        ),
                        "dropout": nn.Dropout(0.1),
                    }
                ),
                "encoder_decoder": EncoderDecoderTransformer(config),
            }
        )

    def forward(
        self: t.Self,
        input_ids: torch.LongTensor,
        output_ids: torch.LongTensor,
        input_masks: torch.LongTensor,
        output_masks: torch.LongTensor,
    ) -> torch.FloatTensor:
        # ids/masks shape: [batch_size, context_length]

        # create input embeddings for tokens and pass through transformer
        inputs = self.model["input"]["dropout"](self.model["input"]["emb"](input_ids))
        outputs = self.model["output"]["dropout"](
            self.model["output"]["emb"](output_ids)
        )
        # inputs/outputs shape: [batch_size, context_length, model_dim]

        # pass inputs, outputs and their masks through encoder-decoder
        hidden = self.model["encoder_decoder"](
            inputs=inputs,
            outputs=outputs,
            input_masks=input_masks,
            output_masks=output_masks,
        )
        # hidden shape: [batch_size, context_length, model_dim]

        # project back to outpu vocabulary size reusing embedding weight matrix (weight-tied)
        unemb = self.model["output"]["emb"].unembed(hidden)
        return nn.functional.log_softmax(unemb, dim=-1)
        # unemb/output shape: [batch_size, context_length, output_vocab_size]

    def configure_optimizers(self: t.Self) -> torch.optim.Optimizer:
        # TODO: use the same learning rate schedule as Attention Is All You Need
        return torch.optim.SGD(self.model.parameters(), lr=3e-4)

    def step(
        self: t.Self, batch: tuple[torch.LongTensor, ...], *, stage: str
    ) -> torch.FloatTensor:
        input_ids, output_ids, target_ids, input_masks, output_masks = batch
        # make predictions
        preds = self(input_ids, output_ids, input_masks, output_masks)
        # flatten to one long sequence and ignore padding in predictions/targets
        output_masks = output_masks.flatten().bool()
        preds = preds.flatten(end_dim=1)[output_masks]
        target_ids = target_ids.flatten()[output_masks]
        # calculate loss
        loss = nn.functional.nll_loss(preds, target_ids)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(
        self: t.Self, batch: tuple[torch.LongTensor, ...]
    ) -> torch.FloatTensor:
        return self.step(batch, stage="train")

    def validation_step(
        self: t.Self, batch: tuple[torch.LongTensor, ...]
    ) -> torch.FloatTensor:
        return self.step(batch, stage="val")

    def test_step(
        self: t.Self, batch: tuple[torch.LongTensor, ...]
    ) -> torch.FloatTensor:
        return self.step(batch, stage="test")

    # def predict_step(
    #     self: t.Self, batch: tuple[torch.LongTensor, ...]
    # ) -> torch.FloatTensor:
    #     ids, targets, masks = batch
    #     preds = self(ids, masks).argmax(axis=-1)
    #     return list(
    #         zip(
    #             *[
    #                 self.tokenizer.batch_decode(
    #                     outputs,
    #                     skip_special_tokens=True,
    #                     clean_up_tokenization_spaces=True,
    #                 )
    #                 for outputs in (preds, targets)
    #             ]
    #         )
    #     )

    # TODO @eonu: rename to 'convert' or something
    # def generate(self: t.Self, string: str | None = None) -> str:
    #     # encode input
    #     tokens = self.tokenizer(
    #         string or "",
    #         add_special_tokens=True,
    #         padding="max_length",
    #         max_length=(self.config.context_length + 1),
    #         return_tensors="pt",
    #     )
    #     ids, mask = itemgetter("input_ids", "attention_mask")(tokens)
    #     ids, mask = (
    #         ids[:, :-1],
    #         mask[:, :-1],
    #     )  # shift input and mask to skip <eos> token
    #     length = mask.sum()

    #     i = 0
    #     while (
    #         i < self.config.context_length - length
    #         and ids[0, -1] != self.tokenizer.eos_token_id
    #     ):
    #         # make prediction
    #         pred = self(ids, mask)

    #         # select first valid ID from 4 most likely IDs for last output
    #         next_id = [
    #             token_id
    #             for token_id in pred[0, -1].topk(4, dim=-1).indices
    #             if token_id
    #             not in (
    #                 self.tokenizer.bos_token_id,
    #                 self.tokenizer.pad_token_id,
    #                 self.tokenizer.unk_token_id,
    #             )
    #         ][0]

    #         # extend ids and mask tensors with new prediction
    #         ids[0, :-1] = ids[0, 1:].clone()
    #         ids[0, -1] = next_id
    #         mask[0, :-1] = mask[0, 1:].clone()
    #         mask[0, -1] = 1

    #         i += 1

    #     # decode IDs and untokenize back to string
    #     output = self.tokenizer.decode(
    #         ids[0].tolist(),
    #         skip_special_tokens=True,
    #         clean_up_tokenization_spaces=True,
    #     )

    #     if ids[0, -1] != self.tokenizer.eos_token_id:
    #         # didn't terminate, add ellipsis at end to indicate
    #         output = f"{output}..."

    #     return output

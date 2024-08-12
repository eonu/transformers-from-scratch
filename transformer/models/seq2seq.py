from __future__ import annotations

import typing as t

import torch
import pydantic as pyd
from torch import nn
from transformers import PreTrainedTokenizer

from transformer.models.base import BaseLM
from transformer.modules.transformers.encoder_decoder import EncoderDecoderTransformer
from transformer.modules.embedding import InputEmbedding
from transformer.params import TransformerParams

__all__ = ["Seq2SeqLM"]


class Seq2SeqLM(BaseLM):
    @pyd.validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self: t.Self,
        config: TransformerParams,
        input_tokenizer: PreTrainedTokenizer,
        output_tokenizer: PreTrainedTokenizer,
    ) -> None:
        super().__init__(config=config)
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        self.model = nn.ModuleDict(
            {
                "input": nn.Sequential(
                    InputEmbedding(len(self.input_tokenizer), config.model_dim),
                    nn.Dropout(0.1),
                ),
                "output": nn.Sequential(
                    InputEmbedding(len(self.input_tokenizer), config.model_dim),
                    nn.Dropout(0.1),
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
        inputs = self.model["input"](input_ids)
        outputs = self.model["output"](output_ids)
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
        unemb = self.model["output"][0].unembed(hidden)
        return nn.functional.log_softmax(unemb, dim=-1)
        # unemb/output shape: [batch_size, context_length, output_vocab_size]

    def configure_optimizers(self: t.Self) -> torch.optim.Optimizer:
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

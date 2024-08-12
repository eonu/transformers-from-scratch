from __future__ import annotations

import typing as t

import torch
import pydantic as pyd
from torch import nn
from transformers import PreTrainedTokenizer

from transformer.models.base import BaseLM
from transformer.modules.transformers.decoder_only import DecoderTransformer
from transformer.modules.embedding import InputEmbedding
from transformer.params import TransformerParams

__all__ = ["CausalLM"]


class CausalLM(BaseLM):
    @pyd.validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self: t.Self,
        config: TransformerParams,
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        super().__init__(config=config)
        self.tokenizer = tokenizer
        self.model = nn.ModuleDict(
            {
                "input": nn.Sequential(
                    InputEmbedding(len(self.tokenizer), config.model_dim),
                    nn.Dropout(0.1),
                ),
                "decoder": DecoderTransformer(config),
            }
        )

    def forward(
        self: t.Self, ids: torch.LongTensor, masks: torch.LongTensor
    ) -> torch.FloatTensor:
        # ids/masks shape: [batch_size, context_length]

        # create input embeddings for tokens and pass through transformer
        emb = self.model["input"](ids)
        hidden = self.model["decoder"](emb, masks=masks)
        # emb/hidden shape: [batch_size, context_length, model_dim]

        # project back to vocabulary size reusing embedding weight matrix (weight-tied)
        unemb = self.model["input"][0].unembed(hidden)
        return nn.functional.log_softmax(unemb, dim=-1)
        # unemb/output shape: [batch_size, context_length, vocab_size]

    def configure_optimizers(self: t.Self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=3e-4)

    def step(
        self: t.Self, batch: tuple[torch.LongTensor, ...], *, stage: str
    ) -> torch.FloatTensor:
        ids, target_ids, masks = batch
        # make predictions
        preds = self(ids, masks)
        # flatten to one long sequence and ignore padding in predictions/targets
        masks = masks.flatten().bool()
        preds = preds.flatten(end_dim=1)[masks]
        target_ids = target_ids.flatten()[masks]
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

    def predict_step(
        self: t.Self, batch: tuple[torch.LongTensor, ...]
    ) -> torch.FloatTensor:
        ids, targets, masks = batch
        preds = self(ids, masks).argmax(axis=-1)
        return list(
            zip(
                *[
                    self.tokenizer.batch_decode(
                        outputs,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                    for outputs in (preds, targets)
                ]
            )
        )

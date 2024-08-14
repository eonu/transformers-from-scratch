from __future__ import annotations

import typing as t

import torch
import pydantic as pyd
from torch import nn
from transformers import PreTrainedTokenizer

from transformer import utils
from transformer.models.base import BaseLM
from transformer.modules.transformers.encoder_only import TransformerEncoder
from transformer.modules.embedding import InputEmbedding
from transformer.params import TransformerParams

__all__ = ["ClassifierLM"]


class ClassifierLM(BaseLM):
    @pyd.validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self: t.Self,
        config: TransformerParams,
        tokenizer: PreTrainedTokenizer,
        num_classes: pyd.PositiveInt,
    ) -> None:
        super().__init__(config=config)
        self.tokenizer = tokenizer
        self.model = nn.ModuleDict(
            {
                "input": nn.Sequential(
                    InputEmbedding(len(self.tokenizer), config.model_dim),
                    nn.Dropout(0.1),
                ),
                "encoder": TransformerEncoder(config),
                "mean": utils.nn.MaskedMean(),
                "softmax": nn.Sequential(
                    nn.Linear(config.model_dim, num_classes),
                    nn.Tanh(),
                    nn.LogSoftmax(dim=-1),
                ),
            }
        )

    def forward(
        self: t.Self, ids: torch.LongTensor, masks: torch.LongTensor
    ) -> torch.FloatTensor:
        # ids/masks shape: [batch_size, context_length]

        # create input embeddings for tokens and pass through transformer
        emb = self.model["input"](ids)
        hidden = self.model["encoder"](emb, masks=masks)
        # emb/hidden shape: [batch_size, context_length, model_dim]

        # calculate the avg. embedding for each sequence (ignoring padding)
        avg = self.model["mean"](hidden, masks=masks)
        # avg shape: [batch_size, model_dim]

        # calculate softmax over averaged encoder output (passed through a linear layer)
        return self.model["softmax"](avg)
        # output shape: [batch_size, num_classes]

    def configure_optimizers(self: t.Self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=3e-4)

    def step(
        self: t.Self, batch: tuple[torch.LongTensor, ...], *, stage: str
    ) -> torch.FloatTensor:
        ids, targets, weights, masks = batch
        # make predictions
        preds = self(ids, masks)
        # calculate (weighted) loss
        loss = nn.functional.nll_loss(preds, targets, reduction="none")
        loss = (weights * loss).mean()
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
                self.tokenizer.batch_decode(
                    ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ),
                preds.tolist(),
                targets.tolist(),
            )
        )

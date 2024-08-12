from __future__ import annotations

import typing as t

import torch
import pydantic as pyd
from torch import nn
from transformers import PreTrainedTokenizer

from transformer import utils
from transformer.models.base import BaseLM
from transformer.modules.transformers.encoder_only import EncoderTransformer
from transformer.modules.embedding import InputEmbedding
from transformer.params import TransformerParams

__all__ = ["RegressorLM"]


class RegressorLM(BaseLM):
    @pyd.validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self: t.Self, config: TransformerParams, tokenizer: PreTrainedTokenizer
    ) -> None:
        super().__init__(config=config)
        self.tokenizer = tokenizer
        self.model = nn.ModuleDict(
            {
                "input": nn.Sequential(
                    InputEmbedding(len(self.tokenizer), config.model_dim),
                    nn.Dropout(0.1),
                ),
                "encoder": EncoderTransformer(config),
                "output": nn.Sequential(
                    nn.Linear(config.model_dim, 1),
                    nn.Sigmoid(),
                    nn.Flatten(),
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
        avg = utils.masked_mean(hidden, masks=masks)
        # avg shape: [batch_size, model_dim]

        # calculate scores over averaged encoder output (passed through a linear layer)
        return self.model["output"](avg)
        # output shape: [batch_size]

    def configure_optimizers(self: t.Self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=3e-4)

    def step(
        self: t.Self, batch: tuple[torch.LongTensor, ...], *, stage: str
    ) -> torch.FloatTensor:
        ids, targets, masks = batch
        # make predictions
        preds = self(ids, masks)
        # calculate loss
        loss = nn.functional.mse_loss(preds, targets)
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
        preds = self(ids, masks)
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

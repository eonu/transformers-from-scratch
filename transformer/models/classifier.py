from __future__ import annotations

import typing as t
from operator import itemgetter

from transformer.modules.transformers.encoder_only import EncoderTransformer
from transformer.modules.embedding import InputEmbedding
from transformer.params import TransformerParams



import torch
import pydantic as pyd
from torch import nn
from lightning import LightningModule
from transformers import PreTrainedTokenizer

class ClassifierLM(LightningModule):
    def __init__(
        self: t.Self,
        config: TransformerParams,
        tokenizer: PreTrainedTokenizer,
        num_classes: pyd.PositiveInt,
    ) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.model = nn.ModuleDict(
            {
                "embedding": InputEmbedding(len(tokenizer), config.model_dim),
                "dropout": nn.Dropout(0.1),
                "encoder": EncoderTransformer(config),
                "softmax": nn.Sequential(
                    nn.AvgPool2d(kernel_size=(config.context_length, 1)),
                    nn.Flatten(start_dim=1),
                    nn.Linear(config.model_dim, num_classes),
                    nn.Tanh(),
                    nn.LogSoftmax(dim=-1),
                )
            }
        )

    def forward(
        self: t.Self, ids: torch.LongTensor, masks: torch.LongTensor
    ) -> torch.FloatTensor:
        # ids/masks shape: [batch_size, context_length]

        # create input embeddings for tokens and pass through transformer
        emb = self.model["dropout"](self.model["embedding"](ids))
        hidden = self.model["encoder"](emb, masks=masks)
        # emb/hidden shape: [batch_size, context_length, model_dim]

        # calculate softmax over averaged encoder output (passed through a linear layer)
        return self.model["softmax"](hidden)
        # output shape: [batch_size, num_classes]

    def configure_optimizers(self: t.Self) -> torch.optim.Optimizer:
        # TODO: use the same learning rate schedule as Attention Is All You Need
        return torch.optim.SGD(self.model.parameters(), lr=3e-4)

    def step(
        self: t.Self, batch: tuple[torch.LongTensor, ...], *, stage: str
    ) -> torch.FloatTensor:
        ids, targets, masks = batch
        # make predictions
        preds = self(ids, masks)
        # calculate loss
        loss = nn.functional.nll_loss(preds, targets)
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
        self: t.Self,
        batch: tuple[torch.LongTensor, ...],
    ) -> torch.FloatTensor:
        return self.step(batch, stage="test")

    def predict_step(
        self: t.Self,
        batch: tuple[torch.LongTensor, ...],
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
                preds,
                targets,
            )
        )

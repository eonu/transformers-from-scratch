from __future__ import annotations

import typing as t
from operator import itemgetter

from transformer.modules.transformers.decoder_only import DecoderTransformer
from transformer.modules.embedding import InputEmbedding
from transformer.params import TransformerParams

from transformers import PreTrainedTokenizer

import torch
from torch import nn
from lightning import LightningModule


class CausalLM(LightningModule):
    def __init__(
        self: t.Self,
        config: TransformerParams,
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.model = nn.ModuleDict(
            {
                "embedding": InputEmbedding(len(tokenizer), config.model_dim),
                "dropout": nn.Dropout(0.1),
                "decoder": DecoderTransformer(config),
            }
        )

    def forward(
        self: t.Self, ids: torch.LongTensor, masks: torch.LongTensor
    ) -> torch.FloatTensor:
        # ids/masks shape: [batch_size, context_length]

        # create input embeddings for tokens and pass through transformer
        emb = self.model["dropout"](self.model["embedding"](ids))
        hidden = self.model["decoder"](emb, masks=masks)
        # emb/hidden shape: [batch_size, context_length, model_dim]

        # project back to vocabulary size reusing embedding weight matrix (weight-tied)
        unemb = self.model["embedding"].unembed(hidden)
        return nn.functional.log_softmax(unemb, dim=-1)
        # unemb/output shape: [batch_size, context_length, vocab_size]

    def configure_optimizers(self: t.Self) -> torch.optim.Optimizer:
        # TODO: use the same learning rate schedule as Attention Is All You Need
        return torch.optim.SGD(self.model.parameters(), lr=3e-4)

    def step(
        self: t.Self, batch: tuple[torch.LongTensor, ...], *, stage: str
    ) -> torch.FloatTensor:
        ids, targets, masks = batch
        # make predictions
        preds = self(ids, masks)
        # flatten to one long sequence and ignore padding in predictions/targets
        masks = masks.flatten().bool()
        preds = preds.flatten(end_dim=1)[masks]
        targets = targets.flatten()[masks]
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
        self: CausalLM,
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

    def generate(self: t.Self, string: str | None = None) -> str:
        # encode input
        tokens = self.tokenizer(
            string or "",
            add_special_tokens=True,
            padding="max_length",
            max_length=(self.config.context_length + 1),
            return_tensors="pt",
        )
        ids, mask = itemgetter("input_ids", "attention_mask")(tokens)
        ids, mask = (
            ids[:, :-1],
            mask[:, :-1],
        )  # shift input and mask to skip <eos> token
        length = mask.sum()

        i = 0
        while (
            i < self.config.context_length - length
            and ids[0, -1] != self.tokenizer.eos_token_id
        ):
            # make prediction
            pred = self(ids, mask)

            # select first valid ID from 4 most likely IDs for last output
            next_id = [
                token_id
                for token_id in pred[0, -1].topk(4, dim=-1).indices
                if token_id
                not in (
                    self.tokenizer.bos_token_id,
                    self.tokenizer.pad_token_id,
                    self.tokenizer.unk_token_id,
                )
            ][0]

            # extend ids and mask tensors with new prediction
            ids[0, :-1] = ids[0, 1:].clone()
            ids[0, -1] = next_id
            mask[0, :-1] = mask[0, 1:].clone()
            mask[0, -1] = 1

            i += 1

        # decode IDs and untokenize back to string
        output = self.tokenizer.decode(
            ids[0].tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        if ids[0, -1] != self.tokenizer.eos_token_id:
            # didn't terminate, add ellipsis at end to indicate
            output = f"{output}..."

        return output

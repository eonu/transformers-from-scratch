from __future__ import annotations

from transformer.modules.transformers.decoder_only import Transformer
from transformer.modules.embedding import InputEmbedding
from transformer.dataloaders.teacher_forcing import TeacherForcingDataModule
from transformer.params import TransformerParams

from transformers import LlamaTokenizerFast

from torch import nn
from lightning import LightningModule, Trainer
import pydantic as pyd
import torch
import pandas as pd


class SamplingTransformer(LightningModule):
    @pyd.validate_call
    def __init__(self, config: TransformerParams, vocab_size: pyd.PositiveInt):
        super().__init__()
        self.config = config
        self.model = nn.ModuleDict(
            {
                "embedding": nn.Sequential(
                    InputEmbedding(vocab_size, config.model_dim),
                    nn.Dropout(0.1),
                ),
                "transformer": Transformer(config),
            }
        )

    def forward(
        self: SamplingTransformer, ids: torch.LongTensor, masks: torch.LongTensor
    ) -> SamplingTransformer:
        # ids shape: [batch_size, context_length]

        # create input embeddings for tokens and pass through transformer
        emb = self.model["embedding"](ids)
        hidden = self.model["transformer"](emb, masks=masks)
        # emb/hidden shape: [batch_size, context_length, model_dim]

        # project back to vocabulary size reusing embedding weight matrix (weight-tied)
        unemb = self.model["embedding"].unembed(hidden)
        return nn.functional.log_softmax(unemb, dim=-1)
        # unemb/output shape: [batch_size, context_length, vocab_size]

    def training_step(self, batch):
        ids, targets, masks = batch
        preds = self(ids, masks)
        # flatten [batch_size, context_length] to one long sequence and compute loss
        loss = nn.functional.nll_loss(preds.flatten(end_dim=1), targets.flatten())
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # TODO: use the same learning rate schedule as Attention Is All You Need
        return torch.optim.SGD(self.model.parameters(), lr=3e-4)


class FloridaManDataModule(TeacherForcingDataModule):
    def setup(self, stage: str):
        # read titles with 200 or fewer characters from CSV
        titles = pd.read_csv("data/florida_man.csv").title
        self.data = titles.loc[titles.str.len() <= 200].tolist()
        super().setup(stage=stage)


if __name__ == "__main__":
    # initialize pretrained tokenizer for causal language modelling
    # - llama does not add an EOS token by default, so override this
    # - llama also does not use a padding token, so this needs to be added
    tokenizer = LlamaTokenizerFast.from_pretrained(
        "huggyllama/llama-7b", add_eos_token=True, from_slow=True
    )
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # initialize the transformer
    model = SamplingTransformer(
        config=TransformerParams(
            num_blocks=6, model_dim=32, feed_forward_dim=64, num_heads=2
        ),
        vocab_size=len(tokenizer),
    )

    # tokenize & encode data and prepare train/test splits
    datamodule = FloridaManDataModule(
        tokenizer=tokenizer,
        context_length=64,
        batch_size=32,
        test_size=0.2,
        limit=100,
        random_state=1,
    )

    # train the model
    trainer = Trainer(max_epochs=50)
    trainer.fit(model=model, datamodule=datamodule)

    breakpoint()
    pass

from __future__ import annotations

from operator import itemgetter

from transformer.modules.transformers.decoder_only import DecoderTransformer
from transformer.modules.embedding import InputEmbedding
from transformer.dataloaders.teacher_forcing import TeacherForcingDataModule
from transformer.params import TransformerParams

from transformers import LlamaTokenizerFast, PreTrainedTokenizer

from torch import nn
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torch
import pandas as pd


class SamplingTransformer(LightningModule):
    def __init__(
        self: SamplingTransformer,
        config: TransformerParams,
        tokenizer: PreTrainedTokenizer,
    ) -> SamplingTransformer:
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
        self: SamplingTransformer, ids: torch.LongTensor, masks: torch.LongTensor
    ) -> SamplingTransformer:
        # ids/masks shape: [batch_size, context_length]

        # create input embeddings for tokens and pass through transformer
        emb = self.model["dropout"](self.model["embedding"](ids))
        hidden = self.model["decoder"](emb, masks=masks)
        # emb/hidden shape: [batch_size, context_length, model_dim]

        # project back to vocabulary size reusing embedding weight matrix (weight-tied)
        unemb = self.model["embedding"].unembed(hidden)
        return nn.functional.log_softmax(unemb, dim=-1)
        # unemb/output shape: [batch_size, context_length, vocab_size]

    def configure_optimizers(self: SamplingTransformer) -> torch.optim.Optimizer:
        # TODO: use the same learning rate schedule as Attention Is All You Need
        return torch.optim.SGD(self.model.parameters(), lr=3e-4)

    def step(
        self: SamplingTransformer, batch: tuple[torch.LongTensor, ...], *, stage: str
    ) -> torch.FloatTensor:
        ids, targets, masks = batch
        preds = self(ids, masks)
        loss = nn.functional.nll_loss(preds.flatten(end_dim=1), targets.flatten())
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(
        self: SamplingTransformer, batch: tuple[torch.LongTensor, ...]
    ) -> torch.FloatTensor:
        return self.step(batch, stage="train")

    def validation_step(
        self: SamplingTransformer, batch: tuple[torch.LongTensor, ...]
    ) -> torch.FloatTensor:
        return self.step(batch, stage="val")

    def test_step(
        self: SamplingTransformer,
        batch: tuple[torch.LongTensor, ...],
    ) -> torch.FloatTensor:
        return self.step(batch, stage="test")

    def predict_step(
        self: SamplingTransformer,
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

    def generate(self, string: str | None = None) -> str:
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


class FloridaManDataModule(TeacherForcingDataModule):
    def setup(self: FloridaManDataModule, stage: str) -> None:
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
    context_length = 64
    model = SamplingTransformer(
        config=TransformerParams(context_length=context_length),
        tokenizer=tokenizer,
    )

    # tokenize & encode data and prepare train/test splits
    datamodule = FloridaManDataModule(
        tokenizer=tokenizer,
        context_length=context_length,
        batch_size=32,
        val_size=0.2,
        test_size=0.1,
        num_workers=9,
        persistent_workers=True,
        limit=None,
        random_state=1,
    )

    # train the model
    trainer = Trainer(
        max_epochs=1000,
        callbacks=EarlyStopping(monitor="val_loss", mode="min", patience=5),
        accelerator="cpu",
    )
    trainer.fit(model=model, datamodule=datamodule)

    # calculate test metrics
    trainer.test(model=model, datamodule=datamodule)

    # view first batch of test set predictions
    # note: these are still produced using teacher-forcing, so not purely generated
    pred = trainer.predict(model=model, datamodule=datamodule)

    output = model.generate("Florida man")
    breakpoint()
    pass

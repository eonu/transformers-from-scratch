from __future__ import annotations

from transformer.params import TransformerParams
from transformer.models.causal import CausalLM
from transformer.dataloaders.teacher_forcing import TeacherForcingDataModule

from transformers import LlamaTokenizerFast

import pandas as pd
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


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
    model = CausalLM(
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
        max_epochs=1,
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

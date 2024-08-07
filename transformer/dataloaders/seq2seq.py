import abc
import typing as t
from operator import itemgetter

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer
from sklearn.model_selection import train_test_split
from lightning import LightningDataModule

__all__ = ["Seq2SeqDataModule", "Seq2SeqDataset"]


class Seq2SeqDataModule(LightningDataModule):
    def __init__(
        self: t.Self,
        input_tokenizer: PreTrainedTokenizer,
        output_tokenizer: PreTrainedTokenizer,
        context_length: int,
        batch_size: int,
        val_size: int,
        test_size: float,
        num_workers: int | None = None,
        persistent_workers: bool = False,
        limit: int | None = None,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        super().__init__()
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        self.context_length = context_length
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.limit = limit
        self.random_state = random_state
        self.data: list[tuple[str, str]] = []
        self.splits: dict[t.Literal["train", "val", "test"], Dataset] = {}

    @abc.abstractmethod
    def setup(self: t.Self, stage: str) -> None:
        # limit the data if specified
        data = self.data[: self.limit] if self.limit else self.data

        # generate train/val/test set splits
        splits: dict[t.Literal["train", "val", "test"], list[str]] = {}
        splits["train"], rest = train_test_split(
            data,
            test_size=(self.val_size + self.test_size),
            random_state=self.random_state,
        )
        splits["val"], splits["test"] = train_test_split(
            rest,
            test_size=(self.test_size / (self.val_size + self.test_size)),
            random_state=self.random_state,
        )

        for split, split_data in splits.items():
            # divide input and output strings into separate lists
            inputs: list[str] = []
            outputs: list[str] = []
            for input_string, output_string in split_data:
                inputs.append(input_string)
                outputs.append(output_string)
            # encode input data and obtain IDs and masks
            input_ids, input_masks = self.encode_inputs(inputs)
            # encode output data and obtain IDs and masks
            output_ids, output_masks = self.encode_outputs(outputs)
            # build dataset
            self.splits[split] = Seq2SeqDataset(
                input_ids=input_ids,
                output_ids=output_ids[:, :-1],
                target_ids=output_ids[:, 1:],
                input_masks=input_masks,
                output_masks=output_masks[:, :-1],
            )

    def dataloader(
        self: t.Self, split: t.Literal["train", "val", "test"], *, shuffle: bool
    ) -> DataLoader:
        return DataLoader(
            self.splits[split],
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def train_dataloader(self: t.Self) -> DataLoader:
        return self.dataloader("train", shuffle=True)

    def val_dataloader(self: t.Self) -> DataLoader:
        return self.dataloader("val", shuffle=False)

    def test_dataloader(self: t.Self) -> DataLoader:
        return self.dataloader("test", shuffle=False)

    def predict_dataloader(self: t.Self) -> DataLoader:
        return self.dataloader("test", shuffle=False)

    def encode_inputs(
        self: t.Self, data: list[str]
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        return self._encode(data, tokenizer=self.input_tokenizer)

    def encode_outputs(
        self: t.Self, data: list[str]
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        return self._encode(
            data, tokenizer=self.output_tokenizer, is_teacher_forcing=True
        )

    def _encode(
        self: t.Self,
        data: list[str],
        *,
        tokenizer: PreTrainedTokenizer,
        is_teacher_forcing: bool = False
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        return itemgetter("input_ids", "attention_mask")(
            tokenizer(
                data,
                padding="max_length",
                truncation=True,
                max_length=(self.context_length + is_teacher_forcing),
                return_tensors="pt",
            )
        )


class Seq2SeqDataset(Dataset):
    """TeacherForcing for outputs, but with inputs also"""

    def __init__(
        self: t.Self,
        input_ids: torch.LongTensor,
        output_ids: torch.LongTensor,
        target_ids: torch.LongTensor,
        input_masks: torch.LongTensor,
        output_masks: torch.LongTensor,
    ) -> None:
        super().__init__()
        self.input_ids = input_ids
        self.output_ids = output_ids
        self.target_ids = target_ids
        self.input_masks = input_masks
        self.output_masks = output_masks

    def __len__(self: t.Self) -> int:
        return len(self.input_ids)

    def __getitem__(self: t.Self, index: int) -> tuple[torch.LongTensor, ...]:
        return (
            self.input_ids[index],
            self.output_ids[index],
            self.target_ids[index],
            self.input_masks[index],
            self.output_masks[index],
        )

import abc
import typing as t
from operator import itemgetter

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer
from sklearn.model_selection import train_test_split
from lightning import LightningDataModule

__all__ = ["CausalDataModule", "CausalDataset"]


class CausalDataModule(LightningDataModule):
    def __init__(
        self: t.Self,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        batch_size: int,
        val_size: float,
        test_size: float,
        num_workers: int | None = None,
        persistent_workers: bool = False,
        limit: int | None = None,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.limit = limit
        self.random_state = random_state
        self.data: list[str] = []
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
            # encode data and obtain examples, labels and masks
            ids, masks = self.encode(split_data)
            self.splits[split] = CausalDataset(
                input_ids=ids[:, :-1],
                target_ids=ids[:, 1:],
                masks=masks[:, :-1],
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

    def encode(
        self: t.Self, data: list[str]
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        return itemgetter("input_ids", "attention_mask")(
            self.tokenizer(
                data,
                padding="max_length",
                truncation=True,
                max_length=(self.context_length + 1),
                return_tensors="pt",
            )
        )


class CausalDataset(Dataset):
    def __init__(
        self: t.Self,
        input_ids: torch.LongTensor,
        target_ids: torch.LongTensor,
        masks: torch.LongTensor,
    ) -> None:
        super().__init__()
        self.input_ids = input_ids
        self.target_ids = target_ids
        self.masks = masks

    def __len__(self: t.Self) -> int:
        return len(self.input_ids)

    def __getitem__(self: t.Self, index: int) -> tuple[torch.LongTensor, ...]:
        return self.input_ids[index], self.target_ids[index], self.masks[index]

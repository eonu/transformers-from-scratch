import abc
import typing as t
from operator import itemgetter
from typing import Literal
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

import numpy as np
from transformers import PreTrainedTokenizer
from sklearn.model_selection import train_test_split
from lightning import LightningDataModule

__all__ = ["InferenceDataModule", "InferenceDataset"]


@dataclass
class Split:
    X: list[str] | None = None
    y: torch.LongTensor | None = None


class InferenceDataModule(LightningDataModule):
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
        self.X, self.y = None, None
        self.splits: dict[Literal["train", "val", "test"], Dataset] = {}

    @abc.abstractmethod
    def setup(self: t.Self, stage: str) -> None:
        # limit the data if specified
        X = self.X[: self.limit] if self.limit else self.X
        y = self.y[: self.limit] if self.limit else self.y

        # generate train/val/test set splits
        splits: dict[Literal["train", "val", "test"], Split] = defaultdict(Split)
        train, val, test = itemgetter("train", "val", "test")(splits)
        stratify = y if y.dtype == torch.long else None
        train.X, X, train.y, y = train_test_split(
            X,
            y,
            test_size=(self.val_size + self.test_size),
            random_state=self.random_state,
            stratify=stratify
        )
        stratify = y if y.dtype == torch.long else None
        val.X, test.X, val.y, test.y = train_test_split(
            X,
            y,
            test_size=(self.test_size / (self.val_size + self.test_size)),
            random_state=self.random_state,
            stratify=stratify
        )

        for split, split_data in splits.items():
            # encode data and obtain examples, labels and masks
            ids, masks = self._encode(split_data.X)
            self.splits[split] = InferenceDataset(
                ids=ids, outputs=split_data.y, masks=masks
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

    def _encode(
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


class InferenceDataset(Dataset):
    def __init__(
        self: t.Self,
        ids: torch.LongTensor,
        outputs: torch.Tensor,
        masks: torch.LongTensor,
    ) -> None:
        super().__init__()
        self.ids = ids
        self.outputs = outputs
        self.masks = masks

    def __len__(self: t.Self) -> int:
        return len(self.ids)

    def __getitem__(
        self: t.Self, index: int
    ) -> tuple[torch.LongTensor, torch.Tensor, torch.LongTensor]:
        return (
            self.ids[index],
            self.outputs[index],
            self.masks[index],
        )

import abc
from operator import itemgetter
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset

import numpy as np
from transformers import PreTrainedTokenizer
from sklearn.model_selection import train_test_split
from lightning import LightningDataModule

__all__ = ["TeacherForcingDataModule", "TeacherForcingDataset"]


class TeacherForcingDataModule(LightningDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        batch_size: int,
        val_size: float,
        test_size: float,
        num_workers: int | None = None,
        persistent_workers: bool = False,
        limit: int | None = None,
        random_state: int | np.random.RandomState | None = None,
    ):
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
        self.data = []
        self.splits: dict[Literal["train", "val", "test"], Dataset] = {}

    @abc.abstractmethod
    def setup(self, stage: str):
        # limit the data if specified
        data = self.data[: self.limit] if self.limit else self.data

        # generate train/val/test set splits
        splits: dict[Literal["train", "val", "test"], list[str]] = {}
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
            ids, masks = self._encode(split_data)
            self.splits[split] = TeacherForcingDataset(
                input_ids=ids[:, :-1],
                target_ids=ids[:, 1:],
                attention_masks=masks[:, :-1]
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.splits["train"], 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            persistent_workers=self.persistent_workers,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.splits["val"], 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.splits["test"], 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            persistent_workers=self.persistent_workers,
        )
    
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.splits["test"], 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            persistent_workers=self.persistent_workers,
        )

    def _encode(self, data: list[str]) -> tuple[torch.LongTensor, torch.LongTensor]:
        return itemgetter("input_ids", "attention_mask")(
            self.tokenizer(
                data,
                padding="max_length",
                truncation=True,
                max_length=(self.context_length + 1),
                return_tensors="pt",
            )
        )


class TeacherForcingDataset(Dataset):
    def __init__(
        self,
        input_ids: torch.LongTensor,
        target_ids: torch.LongTensor,
        attention_masks: torch.LongTensor,
    ):
        super().__init__()
        self.input_ids = input_ids
        self.target_ids = target_ids
        self.attention_masks = attention_masks

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        return (
            self.input_ids[index],
            self.target_ids[index],
            self.attention_masks[index],
        )

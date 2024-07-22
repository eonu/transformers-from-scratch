import abc
from operator import itemgetter

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
        test_size: float,
        limit: int | None = None,
        random_state: int | np.random.RandomState | None = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.batch_size = batch_size
        self.test_size = test_size
        self.limit = limit
        self.random_state = random_state

    @abc.abstractmethod
    def setup(self, stage: str):
        # limit the data if specified
        data = self.data[: self.limit] if self.limit else self.data

        # generate train/ test set splits
        train, test = train_test_split(
            data, test_size=self.test_size, random_state=self.random_state
        )

        # encode training data and obtain examples, labels and masks
        train_ids, train_masks = self._encode(train)
        self.train_set = TeacherForcingDataset(
            input_ids=train_ids[:, :-1],
            target_ids=train_ids[:, 1:],
            attention_masks=train_masks[:, :-1],
        )

        # encode test data and obtain examples, labels and masks
        test_ids, test_masks = self._encode(test)
        self.test_set = TeacherForcingDataset(
            input_ids=test_ids[:, :-1],
            target_ids=test_ids[:, 1:],
            attention_masks=test_masks[:, :-1],
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False)

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

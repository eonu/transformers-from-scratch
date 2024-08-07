from __future__ import annotations

import typing as t

from lightning import LightningModule

from transformer.params import TransformerParams

__all__ = ["BaseLM"]


class BaseLM(LightningModule):
    def __init__(self: t.Self, config: TransformerParams) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters(self.config.model_dump())

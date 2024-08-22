from __future__ import annotations

import typing as t

from lightning import LightningModule

from transformer.params import TransformerParams

__all__ = ["BaseLM"]


class BaseLM(LightningModule):
    def __init__(self: t.Self, params: TransformerParams) -> None:
        super().__init__()
        self.params = params
        self.save_hyperparameters(self.params.model_dump())

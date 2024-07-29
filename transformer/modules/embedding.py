from __future__ import annotations

import typing as t

import torch
import pydantic as pyd
from torch import nn
from lightning import LightningModule

__all__ = ["InputEmbedding", "PositionalEncoding"]


class InputEmbedding(LightningModule):
    def __init__(self: t.Self, vocab_size: pyd.PositiveInt, embedding_size: pyd.PositiveInt) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.word_embedder = nn.Embedding(vocab_size, embedding_size)
        self.position_encoder = PositionalEncoding(embedding_size)

    def forward(self: t.Self, x: torch.LongTensor) -> torch.FloatTensor:
        # x shape: [batch_size, context_length]
        word_emb = self.word_embedder(x)
        positional_enc = self.position_encoder(x)
        return word_emb + positional_enc
        # output shape: [batch_size, context_length, embedding_size]

    def unembed(self: t.Self, x: torch.FloatTensor) -> torch.FloatTensor:
        # x shape: [batch_size, context_length, embedding_size]
        return x @ self.word_embedder.weight.T
        # output shape: [batch_size, context_length, vocab_size]


class PositionalEncoding(LightningModule):
    def __init__(self: t.Self, embedding_size: pyd.PositiveInt) -> None:
        super().__init__()
        self.embedding_size = embedding_size

    def forward(self: t.Self, x: torch.LongTensor) -> torch.FloatTensor:
        B, C = x.shape
        # shape: [batch_size, context_length]
        enc = torch.zeros(*x.shape, self.embedding_size).to(x)
        # shape: [batch_size, context_length, embedding_size]
        idxs_size = self.embedding_size // 2
        idxs = torch.arange(idxs_size).expand(B, C, -1).to(x)
        # shape: [embedding_size // 2]
        pos = torch.arange(C).expand(B, idxs_size, -1).transpose(1, 2).to(x)
        # shape: [batch_size, context_length]
        enc[:, :, ::2] = torch.sin(pos / 10e3 ** (2 * idxs / self.embedding_size))
        enc[:, :, 1::2] = torch.cos(pos / 10e3 ** (2 * idxs / self.embedding_size))
        return enc
        # shape: [batch_size, context_length, embedding_size]

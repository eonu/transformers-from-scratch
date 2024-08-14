from __future__ import annotations

import typing as t

import torch
from torch import nn
from lightning import LightningModule

from transformer.params import TransformerParams, TransformerBlockParams
from transformer.modules.attention import MultiHeadSelfAttention
from transformer.modules.transformers.encoder_only import TransformerEncoder

__all__ = ["TransformerEncoderDecoder"]


class TransformerEncoderDecoder(LightningModule):
    def __init__(self: t.Self, params: TransformerParams) -> None:
        super().__init__()
        self.params: TransformerParams = params
        self.model = nn.ModuleDict(
            {
                "encoder": TransformerEncoder(params=params),
                "decoder": TransformerDecoder(params=params),
            }
        )

    def forward(
        self: t.Self,
        inputs: torch.FloatTensor,
        outputs: torch.FloatTensor,
        input_masks: torch.LongTensor,
        output_masks: torch.LongTensor,
    ) -> torch.FloatTensor:
        enc = self.model["encoder"](inputs, masks=input_masks)
        # hidden shape: [batch_size, context_length, model_dim]
        return self.model["decoder"](outputs, enc, masks=output_masks)
        # shape: [batch_size, context_length, model_dim]


class TransformerDecoder(LightningModule):
    def __init__(self: t.Self, params: TransformerParams) -> None:
        super().__init__()
        self.params: TransformerParams = params
        self.model = nn.ModuleList(
            [
                DecoderTransformerBlock(self.params.block_params)
                for _ in range(self.params.num_blocks)
            ]
        )

    def forward(
        self: t.Self,
        x: torch.FloatTensor,
        enc: torch.FloatTensor,
        masks: torch.LongTensor,
    ) -> None:
        for block in self.model:
            x = block(x, enc, masks=masks)
        return x


class DecoderTransformerBlock(LightningModule):
    def __init__(self: t.Self, params: TransformerBlockParams) -> None:
        super().__init__()
        self.params: TransformerBlockParams = params
        self.model: nn.ModuleDict = nn.ModuleDict(
            {
                "masked": nn.ModuleDict(
                    {
                        "attn": MultiHeadSelfAttention(
                            self.params.multi_head_params.masked
                        ),
                        "dropout": nn.Dropout(0.1),
                        "norm": nn.LayerNorm(self.params.model_dim),
                    }
                ),
                "unmasked": nn.ModuleDict(
                    {
                        "attn": MultiHeadSelfAttention(
                            self.params.multi_head_params.unmasked
                        ),
                        "dropout": nn.Dropout(0.1),
                        "norm": nn.LayerNorm(self.params.model_dim),
                    }
                ),
                "ffnn": nn.ModuleDict(
                    {
                        "fc": nn.Sequential(
                            nn.Linear(
                                self.params.model_dim, self.params.feed_forward_dim
                            ),
                            nn.ReLU(),
                            nn.Linear(
                                self.params.feed_forward_dim, self.params.model_dim
                            ),
                            nn.Dropout(0.1),
                        ),
                        "norm": nn.LayerNorm(self.params.model_dim),
                    }
                ),
            }
        )

    @property
    def masked(self: t.Self) -> nn.ModuleDict:
        return self.model["masked"]

    @property
    def unmasked(self: t.Self) -> nn.ModuleDict:
        return self.model["unmasked"]

    @property
    def ffnn(self: t.Self) -> nn.ModuleDict:
        return self.model["ffnn"]

    def forward(
        self: t.Self,
        x: torch.FloatTensor,
        enc: torch.FloatTensor,
        masks: torch.LongTensor,
    ) -> torch.FloatTensor:
        # calculate masked attention on outputs
        masked_attn = self.masked["dropout"](
            self.masked["attn"](q=x, k=x, v=x, masks=masks)
        )
        masked_attn = self.masked["norm"](x + masked_attn)
        # shape: [batch_size, context_length, model_dim]

        # calculate unmasked attention using:
        # - masked output attention vectors as queries
        # - encoder output as keys and values
        unmasked_attn = self.unmasked["dropout"](
            self.unmasked["attn"](q=masked_attn, k=enc, v=enc, masks=masks)
        )
        unmasked_attn = self.masked["norm"](masked_attn + unmasked_attn)
        # shape: [batch_size, context_length, model_dim]

        # pass attention vectors through FFNN, add residual and normalize
        hidden = self.ffnn["fc"](unmasked_attn)
        return self.ffnn["norm"](unmasked_attn + hidden)
        # shape: [batch_size, context_length, model_dim]

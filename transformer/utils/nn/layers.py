import typing as t

import torch

from transformer.utils.nn.functional import masked_mean

__all__ = ["MaskedMean"]


class MaskedMean(torch.nn.Module):
    def forward(
        self: t.Self, x: torch.FloatTensor, masks: torch.LongTensor
    ) -> torch.FloatTensor:
        return masked_mean(x, masks=masks)

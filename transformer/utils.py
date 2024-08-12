import torch

__all__ = ["EPS", "masked_mean"]


EPS = torch.finfo().eps
"""The smallest representable number such that `1.0 + eps != 1.0`."""


def masked_mean(x: torch.FloatTensor, *, masks: torch.LongTensor) -> torch.FloatTensor:
    """Calculates an averaged embedding over each sequence, ignoring padding.

    Parameters
    ----------
    x:
        Input tensor.
        Shape: [batch_size, context_length, model_dim]

    masks:
        Mask tensor.
        Shape: [batch_size, context_length]

    Returns
    -------
    Masked mean.
    Shape: [batch_size, model_dim]
    """
    masks = masks.unsqueeze(dim=-1)
    return (x * masks).sum(dim=1) / masks.sum(dim=1)

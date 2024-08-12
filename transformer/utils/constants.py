import torch

__all__ = ["EPS"]


EPS = torch.finfo().eps
"""The smallest representable number such that `1.0 + eps != 1.0`."""

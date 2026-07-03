"""Model-related helper utilities."""

import torch.nn as nn


def count_parameters(model: nn.Module, only_trainable: bool = True) -> int:
    """Count the number of parameters in a model.

    Args:
        model: PyTorch model whose parameters are counted.
        only_trainable: If True, count only parameters that require gradients.

    Returns:
        Total number of (trainable) parameters.
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

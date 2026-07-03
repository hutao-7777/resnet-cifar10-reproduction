"""Shared constants and helpers used across training/evaluation scripts."""

from typing import Callable, Dict

import torch.nn as nn

# CIFAR-10 class names in the canonical order.
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def get_model_registry() -> Dict[str, Callable[..., nn.Module]]:
    """Return a mapping from model name to model builder function.

    The import of model constructors is deferred to avoid circular imports
    between the ``models`` and ``utils`` packages.

    Returns:
        Dictionary mapping supported model names to callables that return a
        ``nn.Module`` instance.
    """
    from models import (
        plainnet20,
        resnet20,
        resnet32,
        resnet44,
        resnet56,
        resnet110,
    )

    return {
        "resnet20": resnet20,
        "resnet32": resnet32,
        "resnet44": resnet44,
        "resnet56": resnet56,
        "resnet110": resnet110,
        "plainnet20": plainnet20,
    }

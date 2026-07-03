"""Neural network models for CIFAR-10/100 reproduction."""

from .resnet import plainnet20, resnet20, resnet32, resnet44, resnet56, resnet110

__all__ = [
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "resnet110",
    "plainnet20",
]

"""
Basic building blocks for ResNet and PlainNet on CIFAR-10.

This module implements the ``BasicBlock`` used in the original ResNet paper
(He et al., 2016) adapted for 32x32 CIFAR images, as well as a ``PlainBlock``
that has the same stacked convolution structure but no residual shortcut.
The PlainBlock is useful for ablation studies that demonstrate the effect of
residual connections.
"""

from typing import Optional

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Residual basic block: two 3x3 convolutions with BN and ReLU.

    The shortcut connection is added before the final ReLU. If the spatial
    resolution or the number of channels changes, ``downsample`` projects the
    identity tensor to match the output shape.

    Args:
        in_planes: Number of input channels.
        planes: Number of output channels.
        stride: Stride for the first convolution (default: 1).
        downsample: Optional projection shortcut (default: None).
    """

    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PlainBlock(nn.Module):
    """Plain (non-residual) basic block for ablation studies.

    Identical to ``BasicBlock`` in terms of convolution, batch normalization
    and activation layout, but the shortcut connection is removed. This allows
    a fair comparison between plain networks and residual networks.

    Args:
        in_planes: Number of input channels.
        planes: Number of output channels.
        stride: Stride for the first convolution (default: 1).
    """

    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

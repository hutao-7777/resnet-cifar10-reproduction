"""
ResNet and PlainNet implementations for CIFAR-10.

Following the original ResNet paper (He et al., 2016), the input stage uses a
3x3 convolution without max-pooling because CIFAR-10 images are only 32x32.
The network body consists of three groups of residual/plain blocks, each with
a doubled number of channels and spatial downsampling by stride=2.
"""

from typing import List, Type, Union

import torch
import torch.nn as nn

from .basic_block import BasicBlock, PlainBlock

BlockType = Union[Type[BasicBlock], Type[PlainBlock]]


class ResNet(nn.Module):
    """Generic ResNet/PlainNet backbone for CIFAR-10.

    Args:
        block: Block class to use (``BasicBlock`` or ``PlainBlock``).
        layers: Number of blocks in each of the three stages.
        num_classes: Number of output classes (default: 10).
    """

    def __init__(
        self,
        block: BlockType,
        layers: List[int],
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        self.in_planes = 16

        # Initial 3x3 convolution for 32x32 inputs (no 7x7 or maxpool).
        self.conv1 = nn.Conv2d(
            3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        # Three stages with 16, 32, 64 filters respectively.
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # Kaiming initialization for all conv and fc layers.
        self._initialize_weights()

    def _make_layer(
        self,
        block: BlockType,
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Build a stage by stacking ``block`` instances.

        The first block may downsample when ``stride != 1`` or when the input
        and output channel counts differ.
        """
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """Apply Kaiming/constant initialization to conv, bn and fc layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet20(num_classes: int = 10) -> ResNet:
    """ResNet-20 for CIFAR-10 (layers=[3,3,3])."""
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)


def resnet32(num_classes: int = 10) -> ResNet:
    """ResNet-32 for CIFAR-10 (layers=[5,5,5])."""
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)


def resnet44(num_classes: int = 10) -> ResNet:
    """ResNet-44 for CIFAR-10 (layers=[7,7,7])."""
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes)


def resnet56(num_classes: int = 10) -> ResNet:
    """ResNet-56 for CIFAR-10 (layers=[9,9,9])."""
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)


def resnet110(num_classes: int = 10) -> ResNet:
    """ResNet-110 for CIFAR-10 (layers=[18,18,18])."""
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes)


def plainnet20(num_classes: int = 10) -> ResNet:
    """PlainNet-20 for CIFAR-10 (same depth as ResNet-20, no shortcuts)."""
    return ResNet(PlainBlock, [3, 3, 3], num_classes=num_classes)

"""CIFAR-10/100 data loading utilities with standard augmentation and Cutout.

The dataset is automatically downloaded to ``./data`` if it is not already
present. Standard normalization values and light data augmentation
(RandomCrop + RandomHorizontalFlip + optional Cutout) are applied to the
training set.
"""

from typing import Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# CIFAR-10 channel-wise mean and standard deviation.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# CIFAR-100 channel-wise mean and standard deviation.
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


class Cutout:
    """Randomly mask out a square patch from an image.

    This augmentation is applied after normalization (i.e. on a torch Tensor)
    as described in the original Cutout paper (DeVries & Taylor, 2017).

    Args:
        length: Side length of the square mask in pixels.
    """

    def __init__(self, length: int = 16) -> None:
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply Cutout to a single image tensor.

        Args:
            img: Image tensor of shape ``(C, H, W)``.

        Returns:
            The image tensor with a randomly located square zeroed out.
        """
        _, h, w = img.shape
        mask = np.ones((h, w), dtype=np.float32)

        # Random mask center.
        y = np.random.randint(h)
        x = np.random.randint(w)

        # Mask boundaries (clipped to image bounds).
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask).expand_as(img)
        return img * mask


def _build_transforms(
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    use_cutout: bool = False,
    cutout_length: int = 16,
    train: bool = True,
) -> transforms.Compose:
    """Build a ``Compose`` transform for CIFAR-style datasets.

    Args:
        mean: Channel-wise mean for normalization.
        std: Channel-wise standard deviation for normalization.
        use_cutout: Whether to append Cutout to the training transform.
        cutout_length: Side length of the Cutout mask.
        train: If True, build the training transform; otherwise the test transform.

    Returns:
        A torchvision ``Compose`` transform.
    """
    if train:
        transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        if use_cutout:
            transform_list.append(Cutout(cutout_length))
    else:
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]

    return transforms.Compose(transform_list)


def _build_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    train: bool,
) -> DataLoader:
    """Wrap a dataset in a ``DataLoader`` with sensible defaults."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )


def get_cifar10_loaders(
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: str = "./data",
    use_cutout: bool = False,
    cutout_length: int = 16,
) -> Tuple[DataLoader, DataLoader]:
    """Return CIFAR-10 train and test dataloaders.

    Args:
        batch_size: Mini-batch size for both loaders.
        num_workers: Number of subprocesses used for data loading.
        data_dir: Directory where the dataset will be stored.
        use_cutout: If True, append Cutout to the training transform.
        cutout_length: Side length of the Cutout mask.

    Returns:
        A tuple ``(train_loader, test_loader)``.
    """
    train_transform = _build_transforms(
        CIFAR10_MEAN, CIFAR10_STD, use_cutout, cutout_length, train=True
    )
    test_transform = _build_transforms(CIFAR10_MEAN, CIFAR10_STD, train=False)

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = _build_loader(train_set, batch_size, num_workers, train=True)
    test_loader = _build_loader(test_set, batch_size, num_workers, train=False)

    return train_loader, test_loader


def get_cifar100_loaders(
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: str = "./data",
    use_cutout: bool = False,
    cutout_length: int = 16,
) -> Tuple[DataLoader, DataLoader]:
    """Return CIFAR-100 train and test dataloaders.

    Args:
        batch_size: Mini-batch size for both loaders.
        num_workers: Number of subprocesses used for data loading.
        data_dir: Directory where the dataset will be stored.
        use_cutout: If True, append Cutout to the training transform.
        cutout_length: Side length of the Cutout mask.

    Returns:
        A tuple ``(train_loader, test_loader)``.
    """
    train_transform = _build_transforms(
        CIFAR100_MEAN, CIFAR100_STD, use_cutout, cutout_length, train=True
    )
    test_transform = _build_transforms(CIFAR100_MEAN, CIFAR100_STD, train=False)

    train_set = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_set = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = _build_loader(train_set, batch_size, num_workers, train=True)
    test_loader = _build_loader(test_set, batch_size, num_workers, train=False)

    return train_loader, test_loader

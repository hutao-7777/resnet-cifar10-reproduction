"""Evaluation utilities: top-1 accuracy, per-class accuracy and confusion matrix."""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: Optional[int] = None,
) -> Tuple[float, np.ndarray]:
    """Evaluate a model on the test set.

    Args:
        model: Trained PyTorch model.
        test_loader: Test dataloader.
        device: Device used for inference.
        num_classes: Number of classes. If None, inferred from
            ``test_loader.dataset.classes`` or from the maximum label.

    Returns:
        A tuple containing:
            - Overall top-1 accuracy (percentage).
            - Per-class accuracy array of shape ``(num_classes,)``.
    """
    model.eval()

    if num_classes is None:
        if hasattr(test_loader.dataset, "classes"):
            num_classes = len(test_loader.dataset.classes)
        else:
            # Infer the number of classes from the labels in the loader.
            all_labels: List[int] = []
            for _, targets in test_loader:
                all_labels.extend(targets.numpy().tolist())
            num_classes = max(all_labels) + 1

    class_correct = np.zeros(num_classes, dtype=np.int64)
    class_total = np.zeros(num_classes, dtype=np.int64)
    total_correct = 0
    total_samples = 0

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)

        total_correct += predicted.eq(targets).sum().item()
        total_samples += targets.size(0)

        for label, pred in zip(targets.cpu().numpy(), predicted.cpu().numpy()):
            class_total[label] += 1
            if label == pred:
                class_correct[label] += 1

    overall_acc = 100.0 * total_correct / total_samples
    per_class_acc = 100.0 * class_correct / np.maximum(class_total, 1)

    return overall_acc, per_class_acc


@torch.no_grad()
def get_confusion_matrix(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> np.ndarray:
    """Compute the confusion matrix on the test set.

    Args:
        model: Trained PyTorch model.
        test_loader: Test dataloader.
        device: Device used for inference.
        num_classes: Number of classes.

    Returns:
        Confusion matrix of shape ``(num_classes, num_classes)``.
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy().tolist())
        all_labels.extend(targets.numpy().tolist())

    return confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

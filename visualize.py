"""Visualization utilities for training curves, feature maps, confusion matrix,
learning-rate schedule and misclassified samples.

This script reads training histories saved by ``Trainer`` (``.npz`` files under
``runs/{model_name}/history.npz``) and generates figures under ``results/``.
"""

import argparse
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import (
    CIFAR10_CLASSES,
    get_cifar10_loaders,
    get_cifar100_loaders,
    get_confusion_matrix,
    get_model_registry,
)
from utils.data_loader import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR100_MEAN,
    CIFAR100_STD,
)

RESULTS_DIR = "results"

DATASET_LOADERS = {
    "cifar10": get_cifar10_loaders,
    "cifar100": get_cifar100_loaders,
}

DATASET_STATS = {
    "cifar10": (CIFAR10_MEAN, CIFAR10_STD),
    "cifar100": (CIFAR100_MEAN, CIFAR100_STD),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    model_choices = list(get_model_registry().keys())

    parser = argparse.ArgumentParser(
        description="Generate visualizations for trained models",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint for feature map / confusion matrix plots",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet20",
        choices=model_choices,
        help="Model architecture used for the checkpoint",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=list(DATASET_LOADERS.keys()),
        help="Dataset used for evaluation visualizations",
    )
    parser.add_argument(
        "--resnet_history",
        type=str,
        default=None,
        help="Path to .npz with 'train_losses', 'test_accuracies' and 'learning_rates'",
    )
    parser.add_argument(
        "--plain_history",
        type=str,
        default=None,
        help="Path to .npz with 'train_losses', 'test_accuracies' and 'learning_rates'",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Mini-batch size for the test loader",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory for downloaded datasets",
    )
    return parser.parse_args()


def ensure_results_dir() -> None:
    """Create the directory that will hold the generated figures."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_training_curve(
    train_losses: Optional[List[float]] = None,
    test_accuracies: Optional[List[float]] = None,
    save_path: str = os.path.join(RESULTS_DIR, "training_curve.png"),
) -> None:
    """Plot and save the training loss and test accuracy curves.

    Args:
        train_losses: Per-epoch training losses (optional).
        test_accuracies: Per-epoch test accuracies in percentage (optional).
        save_path: File path where the figure is saved.
    """
    if train_losses is None and test_accuracies is None:
        print("No training history provided; skipping training curve plot")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    if train_losses is not None:
        epochs = range(1, len(train_losses) + 1)
        color = "tab:blue"
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train Loss", color=color)
        ax1.plot(epochs, train_losses, color=color, linewidth=2, label="Train Loss")
        ax1.tick_params(axis="y", labelcolor=color)

    if test_accuracies is not None:
        ax2 = ax1.twinx()
        epochs = range(1, len(test_accuracies) + 1)
        color = "tab:orange"
        ax2.set_ylabel("Test Accuracy (%)", color=color)
        ax2.plot(
            epochs,
            test_accuracies,
            color=color,
            linewidth=2,
            label="Test Accuracy",
        )
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.set_ylim([0, 100])

    fig.tight_layout()
    plt.title("Training Curve")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training curve to {save_path}")


def plot_lr_schedule(
    learning_rates: Optional[List[float]] = None,
    save_path: str = os.path.join(RESULTS_DIR, "lr_schedule.png"),
) -> None:
    """Plot and save the learning-rate schedule across epochs."""
    if learning_rates is None or len(learning_rates) == 0:
        print("No learning rate history provided; skipping lr_schedule.png")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(learning_rates) + 1),
        learning_rates,
        linewidth=2,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved learning rate schedule to {save_path}")


def plot_resnet_vs_plainnet(
    resnet_history_path: Optional[str] = None,
    plain_history_path: Optional[str] = None,
    save_path: str = os.path.join(RESULTS_DIR, "resnet_vs_plainnet.png"),
) -> None:
    """Plot test accuracy comparison between ResNet-20 and PlainNet-20."""
    if resnet_history_path is None or plain_history_path is None:
        print(
            "ResNet/PlainNet history paths not provided; "
            "skipping resnet_vs_plainnet.png"
        )
        return

    resnet_data = np.load(resnet_history_path)
    plain_data = np.load(plain_history_path)
    resnet_acc = resnet_data["test_accuracies"]
    plain_acc = plain_data["test_accuracies"]

    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(resnet_acc) + 1),
        resnet_acc,
        linewidth=2,
        label="ResNet-20",
    )
    plt.plot(
        range(1, len(plain_acc) + 1),
        plain_acc,
        linewidth=2,
        label="PlainNet-20",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("ResNet-20 vs PlainNet-20 on CIFAR-10")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim([0, 100])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved ResNet vs PlainNet comparison to {save_path}")


def plot_feature_maps(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    save_path: str = os.path.join(RESULTS_DIR, "feature_maps.png"),
    num_maps: int = 6,
) -> None:
    """Visualize the first ``num_maps`` feature maps of the first conv layer.

    Args:
        model: Trained PyTorch model containing a ``conv1`` layer.
        test_loader: Test dataloader used to fetch a single sample.
        device: Device used for inference.
        save_path: File path where the figure is saved.
        num_maps: Number of feature maps to visualize (default: 6).
    """
    model.eval()
    images, _ = next(iter(test_loader))
    image = images[0:1].to(device)

    # Hook to capture the output of the first convolutional layer.
    feature_maps: Optional[torch.Tensor] = None

    def hook_fn(
        module: nn.Module,
        _input: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        nonlocal feature_maps
        feature_maps = output.detach()

    handle = model.conv1.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(image)
    handle.remove()

    if feature_maps is None:
        raise RuntimeError("Failed to capture feature maps.")

    maps = feature_maps[0, :num_maps].cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    axes = axes.flatten()
    for i in range(num_maps):
        ax = axes[i]
        ax.imshow(maps[i], cmap="viridis")
        ax.set_title(f"Feature Map {i + 1}")
        ax.axis("off")

    plt.suptitle("First Convolution Layer Feature Maps")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved feature maps to {save_path}")


def plot_confusion_matrix(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    save_path: str = os.path.join(RESULTS_DIR, "confusion_matrix.png"),
) -> None:
    """Plot and save the confusion matrix heatmap."""
    cm = get_confusion_matrix(model, test_loader, device, num_classes=len(class_names))

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix on Test Set")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def _denormalize(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> torch.Tensor:
    """Undo normalization for visualization purposes."""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0.0, 1.0)


def plot_misclassified_samples(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    save_path: str = os.path.join(RESULTS_DIR, "misclassified_samples.png"),
    grid: Tuple[int, int] = (4, 4),
) -> None:
    """Visualize a grid of misclassified test samples.

    Args:
        model: Trained PyTorch model.
        test_loader: Test dataloader.
        device: Device used for inference.
        class_names: List of class names.
        mean: Channel-wise mean used for normalization.
        std: Channel-wise standard deviation used for normalization.
        save_path: File path where the figure is saved.
        grid: Grid dimensions ``(rows, cols)`` for the figure.
    """
    model.eval()
    misclassified: List[Tuple[torch.Tensor, int, int]] = []
    needed = grid[0] * grid[1]

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        _, predicted = outputs.max(1)

        mask = predicted != targets
        for img, true, pred in zip(inputs[mask], targets[mask], predicted[mask]):
            misclassified.append((img.cpu(), true.item(), pred.item()))
            if len(misclassified) >= needed:
                break
        if len(misclassified) >= needed:
            break

    if len(misclassified) == 0:
        print("No misclassified samples found; skipping misclassified_samples.png")
        return

    fig, axes = plt.subplots(grid[0], grid[1], figsize=(12, 12))
    axes = axes.flatten()
    for i in range(needed):
        ax = axes[i]
        if i < len(misclassified):
            img, true_label, pred_label = misclassified[i]
            img = _denormalize(img, mean, std)
            np_img = img.permute(1, 2, 0).numpy()
            ax.imshow(np_img)
            ax.set_title(
                f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}"
            )
        ax.axis("off")

    plt.suptitle("Misclassified Samples (4x4 Grid)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved misclassified samples to {save_path}")


def main() -> None:
    args = parse_args()
    ensure_results_dir()

    # Resolve history paths that actually exist on disk.
    resnet_history = (
        args.resnet_history
        if args.resnet_history is not None and os.path.isfile(args.resnet_history)
        else None
    )
    plain_history = (
        args.plain_history
        if args.plain_history is not None and os.path.isfile(args.plain_history)
        else None
    )

    # 1. Training curve: draw ResNet curve by default; fall back to PlainNet
    # when only PlainNet history is available.
    if resnet_history is not None:
        data = np.load(resnet_history)
        plot_training_curve(
            train_losses=data.get("train_losses", None),
            test_accuracies=data.get("test_accuracies", None),
        )
        plot_lr_schedule(learning_rates=data.get("learning_rates", None))

    if plain_history is not None:
        data = np.load(plain_history)
        save_path = (
            os.path.join(RESULTS_DIR, "training_curve.png")
            if resnet_history is None
            else os.path.join(RESULTS_DIR, "training_curve_plainnet.png")
        )
        plot_training_curve(
            train_losses=data.get("train_losses", None),
            test_accuracies=data.get("test_accuracies", None),
            save_path=save_path,
        )
        # Save PlainNet LR schedule with a distinct name when both are present.
        lr_save_path = (
            os.path.join(RESULTS_DIR, "lr_schedule.png")
            if resnet_history is None
            else os.path.join(RESULTS_DIR, "lr_schedule_plainnet.png")
        )
        plot_lr_schedule(
            learning_rates=data.get("learning_rates", None),
            save_path=lr_save_path,
        )

    # 2. ResNet vs PlainNet comparison.
    plot_resnet_vs_plainnet(resnet_history, plain_history)

    if args.checkpoint is None:
        print(
            "No checkpoint provided; skipping feature map and confusion matrix plots."
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model and checkpoint...")
    num_classes = 10 if args.dataset == "cifar10" else 100
    model = get_model_registry()[args.model](num_classes=num_classes).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Preparing {args.dataset.upper()} test loader...")
    loader_fn = DATASET_LOADERS[args.dataset]
    _, test_loader = loader_fn(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
    )

    class_names = (
        CIFAR10_CLASSES if args.dataset == "cifar10" else test_loader.dataset.classes
    )
    mean, std = DATASET_STATS[args.dataset]

    # 3. Feature maps.
    plot_feature_maps(model, test_loader, device)

    # 4. Confusion matrix.
    plot_confusion_matrix(model, test_loader, device, class_names)

    # 5. Misclassified samples.
    plot_misclassified_samples(model, test_loader, device, class_names, mean, std)


if __name__ == "__main__":
    main()

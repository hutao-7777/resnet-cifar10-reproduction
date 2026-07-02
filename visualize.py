"""
Visualization utilities for training curves, feature maps and confusion matrix.

This script reads training histories saved by ``Trainer`` (or TensorBoard logs)
and generates four figures under ``results/``:
    1. training_curve.png
    2. resnet_vs_plainnet.png
    3. feature_maps.png
    4. confusion_matrix.png
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

from models import plainnet20, resnet20, resnet32
from utils import get_cifar10_loaders, get_confusion_matrix


MODEL_REGISTRY = {
    "resnet20": resnet20,
    "resnet32": resnet32,
    "plainnet20": plainnet20,
}

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

RESULTS_DIR = "results"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
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
        choices=list(MODEL_REGISTRY.keys()),
        help="Model architecture used for the checkpoint",
    )
    parser.add_argument(
        "--resnet_history",
        type=str,
        default=None,
        help="Path to .npz with 'test_accuracies' for ResNet-20",
    )
    parser.add_argument(
        "--plain_history",
        type=str,
        default=None,
        help="Path to .npz with 'test_accuracies' for PlainNet-20",
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
        help="Directory for CIFAR-10 dataset",
    )
    return parser.parse_args()


def ensure_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_training_curve(
    train_losses: Optional[List[float]] = None,
    test_accuracies: Optional[List[float]] = None,
    save_path: str = os.path.join(RESULTS_DIR, "training_curve.png"),
) -> None:
    """Plot and save the training loss and test accuracy curves."""
    if train_losses is None and test_accuracies is None:
        print("No training history provided; skipping training_curve.png")
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
    """Visualize the first ``num_maps`` feature maps of the first conv layer."""
    model.eval()
    images, _ = next(iter(test_loader))
    image = images[0:1].to(device)

    # Hook to capture the output of the first convolutional layer.
    feature_maps: Optional[torch.Tensor] = None

    def hook_fn(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
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
    save_path: str = os.path.join(RESULTS_DIR, "confusion_matrix.png"),
) -> None:
    """Plot and save the confusion matrix heatmap."""
    cm = get_confusion_matrix(model, test_loader, device, num_classes=10)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CIFAR10_CLASSES,
        yticklabels=CIFAR10_CLASSES,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix on CIFAR-10 Test Set")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def main() -> None:
    args = parse_args()
    ensure_results_dir()

    # 1. Training curve (requires manually saved .npz histories).
    if args.resnet_history is not None and os.path.isfile(args.resnet_history):
        data = np.load(args.resnet_history)
        plot_training_curve(
            train_losses=data.get("train_losses", None),
            test_accuracies=data.get("test_accuracies", None),
        )

    # 2. ResNet vs PlainNet comparison.
    plot_resnet_vs_plainnet(args.resnet_history, args.plain_history)

    if args.checkpoint is None:
        print("No checkpoint provided; skipping feature map and confusion matrix plots.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model and checkpoint...")
    model = MODEL_REGISTRY[args.model](num_classes=10).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Preparing CIFAR-10 test loader...")
    _, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
    )

    # 3. Feature maps.
    plot_feature_maps(model, test_loader, device)

    # 4. Confusion matrix.
    plot_confusion_matrix(model, test_loader, device)


if __name__ == "__main__":
    main()

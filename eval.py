"""
Evaluation entry point for a trained checkpoint.

Example:
    python eval.py --checkpoint checkpoints/best.pth --model resnet20
"""

import argparse
import os

import torch

from models import plainnet20, resnet20, resnet32
from utils import evaluate, get_cifar10_loaders


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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a ResNet/PlainNet checkpoint on CIFAR-10",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet20",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model architecture",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Mini-batch size",
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


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

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

    print("\nEvaluating...")
    overall_acc, per_class_acc = evaluate(model, test_loader, device)

    print(f"\nTop-1 Test Accuracy: {overall_acc:.2f}%")
    print("Per-class Accuracy:")
    for cls_name, acc in zip(CIFAR10_CLASSES, per_class_acc):
        print(f"  {cls_name:12s}: {acc:.2f}%")


if __name__ == "__main__":
    main()

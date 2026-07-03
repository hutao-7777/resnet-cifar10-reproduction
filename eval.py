"""Evaluation entry point for a trained checkpoint.

Example:
    python eval.py --checkpoint checkpoints/best.pth --model resnet20 --dataset cifar10
"""

import argparse
import os

import torch

from utils import (
    CIFAR10_CLASSES,
    evaluate,
    get_cifar10_loaders,
    get_cifar100_loaders,
    get_model_registry,
)

DATASET_LOADERS = {
    "cifar10": get_cifar10_loaders,
    "cifar100": get_cifar100_loaders,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    model_choices = list(get_model_registry().keys())

    parser = argparse.ArgumentParser(
        description="Evaluate a ResNet/PlainNet checkpoint on CIFAR-10/100",
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
        choices=model_choices,
        help="Model architecture",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=list(DATASET_LOADERS.keys()),
        help="Dataset used for evaluation",
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
        help="Directory for downloaded datasets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

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

    # Use built-in class names from the dataset; CIFAR-10 also has a hard-coded
    # tuple for consistent formatting.
    class_names = (
        CIFAR10_CLASSES if args.dataset == "cifar10" else test_loader.dataset.classes
    )

    print("\nEvaluating...")
    overall_acc, per_class_acc = evaluate(model, test_loader, device)

    print(f"\nTop-1 Test Accuracy: {overall_acc:.2f}%")
    print("Per-class Accuracy:")
    for cls_name, acc in zip(class_names, per_class_acc):
        print(f"  {cls_name:12s}: {acc:.2f}%")


if __name__ == "__main__":
    main()

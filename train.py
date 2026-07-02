"""
Training entry point for ResNet-CIFAR10 reproduction.

Example:
    python train.py --model resnet20 --epochs 200 --batch_size 128 --lr 0.1
"""

import argparse
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from models import plainnet20, resnet20, resnet32
from utils import get_cifar10_loaders, Trainer


MODEL_REGISTRY = {
    "resnet20": resnet20,
    "resnet32": resnet32,
    "plainnet20": plainnet20,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ResNet/PlainNet on CIFAR-10",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet20",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model architecture to train",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Mini-batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Initial learning rate",
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
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="runs",
        help="Directory for TensorBoard logs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.run_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Preparing CIFAR-10 dataloaders (batch_size={args.batch_size})...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
    )

    print(f"Building model: {args.model}")
    model = MODEL_REGISTRY[args.model](num_classes=10)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    log_dir = os.path.join(args.run_dir, args.model)
    writer = SummaryWriter(log_dir=log_dir)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        writer=writer,
        lr=args.lr,
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
    )

    print(f"\nStarting training for {args.epochs} epochs...")
    trainer.train(args.epochs)

    writer.close()


if __name__ == "__main__":
    main()

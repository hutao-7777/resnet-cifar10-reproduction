"""Training entry point for ResNet-CIFAR10/100 reproduction.

Examples:
    # Command-line only
    python train.py --model resnet20 --epochs 200 --batch_size 128 --lr 0.1

    # YAML config with optional command-line overrides
    python train.py --config configs/resnet20.yaml --epochs 10
"""

import argparse
import os

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from utils import (
    Trainer,
    count_parameters,
    get_cifar10_loaders,
    get_cifar100_loaders,
    get_model_registry,
)

# Supported datasets and their loader functions.
DATASET_LOADERS = {
    "cifar10": get_cifar10_loaders,
    "cifar100": get_cifar100_loaders,
}


def _build_parser() -> argparse.ArgumentParser:
    """Return the argument parser with all training options."""
    model_choices = list(get_model_registry().keys())

    parser = argparse.ArgumentParser(
        description="Train ResNet/PlainNet on CIFAR-10/100",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file (command-line args override it)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet20",
        choices=model_choices,
        help="Model architecture to train",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=list(DATASET_LOADERS.keys()),
        help="Dataset to train on",
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
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "multistep"],
        help="Learning-rate schedule",
    )
    parser.add_argument(
        "--milestones",
        type=int,
        nargs="+",
        default=[82, 123],
        help="Epochs at which MultiStepLR decays the learning rate",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Multiplicative LR decay factor for MultiStepLR",
    )
    parser.add_argument(
        "--use_cutout",
        action="store_true",
        help="Apply Cutout augmentation to the training set",
    )
    parser.add_argument(
        "--cutout_length",
        type=int,
        default=16,
        help="Side length of the Cutout mask",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor for CrossEntropyLoss",
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
        help="Directory for TensorBoard logs and training histories",
    )
    return parser


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments, with optional YAML config as defaults."""
    # First pass: only resolve the --config path so YAML values can be used as
    # defaults before the full parser sees the command-line args.
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file (command-line args override it)",
    )
    config_args, remaining_argv = config_parser.parse_known_args()

    config_defaults = {}
    if config_args.config is not None:
        if not os.path.isfile(config_args.config):
            raise FileNotFoundError(f"Config file not found: {config_args.config}")
        with open(config_args.config, "r", encoding="utf-8") as f:
            config_defaults = yaml.safe_load(f) or {}

    # Second pass: full parser with YAML values as defaults.
    parser = _build_parser()
    parser.set_defaults(**config_defaults)
    return parser.parse_args(remaining_argv)


def main() -> None:
    args = parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.run_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(
        f"Preparing {args.dataset.upper()} dataloaders (batch_size={args.batch_size})..."
    )
    loader_fn = DATASET_LOADERS[args.dataset]
    train_loader, test_loader = loader_fn(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        use_cutout=args.use_cutout,
        cutout_length=args.cutout_length,
    )

    num_classes = len(train_loader.dataset.classes)
    print(f"Building model: {args.model} (num_classes={num_classes})")
    model = get_model_registry()[args.model](num_classes=num_classes)
    total_params = count_parameters(model)
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
        run_dir=args.run_dir,
        model_name=args.model,
        lr_scheduler=args.lr_scheduler,
        milestones=args.milestones,
        gamma=args.gamma,
        label_smoothing=args.label_smoothing,
    )

    print(f"\nStarting training for {args.epochs} epochs...")
    trainer.train(args.epochs)

    writer.close()


if __name__ == "__main__":
    main()

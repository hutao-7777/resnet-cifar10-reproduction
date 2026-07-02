"""
Training loop for CIFAR-10 classification.

The ``Trainer`` class encapsulates a standard SGD optimizer with momentum and
weight decay, cosine annealing learning-rate schedule, and TensorBoard logging.
The best checkpoint (by test accuracy) is saved to ``checkpoints/best.pth``.
"""

import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    """Trainer for ResNet/PlainNet on CIFAR-10.

    Args:
        model: PyTorch model to train.
        train_loader: Training dataloader.
        test_loader: Test dataloader.
        device: Device used for training.
        writer: TensorBoard ``SummaryWriter`` instance.
        lr: Initial learning rate (default: 0.1).
        num_epochs: Total number of epochs for cosine annealing (default: 200).
        checkpoint_dir: Directory where checkpoints are saved (default: "checkpoints").
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        writer: Optional[SummaryWriter] = None,
        lr: float = 0.1,
        num_epochs: int = 200,
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.writer = writer
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs
        )

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.train_losses: List[float] = []
        self.test_accuracies: List[float] = []
        self.best_acc: float = 0.0

    def train_epoch(self) -> float:
        """Train for one epoch and return the average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def eval_epoch(self) -> float:
        """Evaluate on the test set and return accuracy (percentage)."""
        self.model.eval()
        correct = 0
        total = 0

        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        accuracy = 100.0 * correct / total
        return accuracy

    def train(self, num_epochs: int) -> None:
        """Main training loop."""
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch()
            test_acc = self.eval_epoch()
            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.test_accuracies.append(test_acc)

            current_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch:03d}/{num_epochs:03d}] "
                f"Loss: {train_loss:.4f} | Test Acc: {test_acc:.2f}% | LR: {current_lr:.6f}"
            )

            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Accuracy/test", test_acc, epoch)
                self.writer.add_scalar("LearningRate", current_lr, epoch)

            # Save the best checkpoint.
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                best_path = os.path.join(self.checkpoint_dir, "best.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_acc": self.best_acc,
                    },
                    best_path,
                )
                print(f"  -> Saved best checkpoint with accuracy {self.best_acc:.2f}%")

        print(f"Training finished. Best test accuracy: {self.best_acc:.2f}%")

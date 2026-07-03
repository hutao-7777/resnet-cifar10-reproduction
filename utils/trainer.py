"""Training loop for CIFAR-10/100 classification.

The ``Trainer`` class encapsulates a standard SGD optimizer with momentum and
weight decay, a configurable learning-rate schedule (cosine or multistep),
optional label smoothing, TensorBoard logging, and automatic checkpointing.
The best checkpoint (by test accuracy) is saved to ``{checkpoint_dir}/best.pth``
and training history is saved to ``{run_dir}/{model_name}/history.npz`` for
later visualization.
"""

import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    """Trainer for ResNet/PlainNet on CIFAR-10/100.

    Args:
        model: PyTorch model to train.
        train_loader: Training dataloader.
        test_loader: Test dataloader.
        device: Device used for training.
        writer: TensorBoard ``SummaryWriter`` instance (optional).
        lr: Initial learning rate (default: 0.1).
        num_epochs: Total number of epochs for cosine annealing (default: 200).
        checkpoint_dir: Directory where checkpoints are saved
            (default: "checkpoints").
        run_dir: Directory where TensorBoard logs and training histories are
            saved (default: "runs").
        model_name: Identifier for the model, used as a subdirectory name under
            ``run_dir`` (default: None).
        lr_scheduler: Learning-rate schedule, either "cosine" or "multistep"
            (default: "cosine").
        milestones: Epoch indices at which the learning rate is decayed by
            ``gamma`` when ``lr_scheduler="multistep"" (default: [82, 123]).
        gamma: Multiplicative factor for MultiStepLR (default: 0.1).
        label_smoothing: Label smoothing factor for CrossEntropyLoss
            (default: 0.0).
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
        run_dir: str = "runs",
        model_name: Optional[str] = None,
        lr_scheduler: str = "cosine",
        milestones: Optional[List[int]] = None,
        gamma: float = 0.1,
        label_smoothing: float = 0.0,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.writer = writer
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir

        # Directory used to store the per-epoch training history (.npz).
        self.history_dir = os.path.join(run_dir, model_name) if model_name else run_dir
        os.makedirs(self.history_dir, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
        )
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4,
        )
        self.scheduler = self._build_scheduler(
            lr_scheduler=lr_scheduler,
            milestones=milestones,
            gamma=gamma,
        )

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.train_losses: List[float] = []
        self.test_accuracies: List[float] = []
        self.learning_rates: List[float] = []
        self.best_acc: float = 0.0

    def _build_scheduler(
        self,
        lr_scheduler: str,
        milestones: Optional[List[int]],
        gamma: float,
    ) -> optim.lr_scheduler._LRScheduler:
        """Build the learning-rate scheduler based on the selected strategy."""
        if lr_scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.num_epochs
            )
        if lr_scheduler == "multistep":
            return optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones if milestones is not None else [82, 123],
                gamma=gamma,
            )
        raise ValueError(
            f"Unsupported lr_scheduler: {lr_scheduler}. "
            "Choose 'cosine' or 'multistep'."
        )

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

    def _save_history(self) -> None:
        """Persist training losses, accuracies and learning rates to a .npz file."""
        history_path = os.path.join(self.history_dir, "history.npz")
        np.savez(
            history_path,
            train_losses=np.array(self.train_losses, dtype=np.float32),
            test_accuracies=np.array(self.test_accuracies, dtype=np.float32),
            learning_rates=np.array(self.learning_rates, dtype=np.float32),
        )

    def train(self, num_epochs: int) -> None:
        """Run the main training loop.

        Args:
            num_epochs: Number of epochs to train.
        """
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch()
            test_acc = self.eval_epoch()
            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            self.train_losses.append(train_loss)
            self.test_accuracies.append(test_acc)
            self.learning_rates.append(current_lr)
            self._save_history()

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

"""Smoke tests using synthetic CIFAR-like data.

These tests verify that the training, evaluation and visualization pipelines
run end-to-end without requiring a real CIFAR download.
"""

import os
import shutil
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from models import resnet20
from utils import Trainer, count_parameters, evaluate
from utils.data_loader import Cutout
from visualize import (
    plot_confusion_matrix,
    plot_feature_maps,
    plot_lr_schedule,
    plot_misclassified_samples,
    plot_training_curve,
)


@pytest.fixture
def synthetic_loaders():
    """Return small synthetic train/test loaders for 10-class classification."""
    x_train = torch.randn(64, 3, 32, 32)
    y_train = torch.randint(0, 10, (64,))
    x_test = torch.randn(32, 3, 32, 32)
    y_test = torch.randint(0, 10, (32,))

    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=16, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test), batch_size=16, shuffle=False
    )
    return train_loader, test_loader


@pytest.fixture
def temp_dirs():
    """Create temporary checkpoint/run/results directories."""
    tmp = tempfile.mkdtemp()
    dirs = {
        "checkpoint_dir": os.path.join(tmp, "checkpoints"),
        "run_dir": os.path.join(tmp, "runs"),
        "results_dir": os.path.join(tmp, "results"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    yield dirs
    shutil.rmtree(tmp)


def test_count_parameters():
    model = resnet20(num_classes=10)
    params = count_parameters(model)
    assert params > 0


def test_cutout():
    cutout = Cutout(length=16)
    img = torch.ones(3, 32, 32)
    out = cutout(img)
    assert out.shape == (3, 32, 32)
    # At least one pixel should be zeroed.
    assert (out == 0).any()


def test_trainer_cosine(synthetic_loaders, temp_dirs):
    train_loader, test_loader = synthetic_loaders
    device = torch.device("cpu")
    model = resnet20(num_classes=10)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        writer=None,
        lr=0.1,
        num_epochs=2,
        checkpoint_dir=temp_dirs["checkpoint_dir"],
        run_dir=temp_dirs["run_dir"],
        model_name="resnet20",
        lr_scheduler="cosine",
    )
    trainer.train(2)

    assert len(trainer.train_losses) == 2
    assert len(trainer.test_accuracies) == 2
    assert len(trainer.learning_rates) == 2
    assert os.path.isfile(os.path.join(temp_dirs["run_dir"], "resnet20", "history.npz"))
    assert os.path.isfile(os.path.join(temp_dirs["checkpoint_dir"], "best.pth"))


def test_trainer_multistep_label_smoothing(synthetic_loaders, temp_dirs):
    train_loader, test_loader = synthetic_loaders
    device = torch.device("cpu")
    model = resnet20(num_classes=10)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        writer=None,
        lr=0.1,
        num_epochs=2,
        checkpoint_dir=temp_dirs["checkpoint_dir"],
        run_dir=temp_dirs["run_dir"],
        model_name="resnet20_ms",
        lr_scheduler="multistep",
        milestones=[2],
        gamma=0.1,
        label_smoothing=0.1,
    )
    trainer.train(2)

    # Milestone 2 is hit after the second epoch, so the second LR is decayed.
    assert trainer.learning_rates[-1] < trainer.learning_rates[0]


def test_evaluate(synthetic_loaders):
    _, test_loader = synthetic_loaders
    device = torch.device("cpu")
    model = resnet20(num_classes=10).to(device)
    overall_acc, per_class_acc = evaluate(model, test_loader, device)

    assert 0 <= overall_acc <= 100
    assert per_class_acc.shape == (10,)


def test_evaluate_with_explicit_num_classes(synthetic_loaders):
    _, test_loader = synthetic_loaders
    device = torch.device("cpu")
    model = resnet20(num_classes=10).to(device)
    overall_acc, per_class_acc = evaluate(model, test_loader, device, num_classes=10)

    assert 0 <= overall_acc <= 100
    assert per_class_acc.shape == (10,)


def test_visualization_plots(synthetic_loaders, temp_dirs):
    train_loader, test_loader = synthetic_loaders
    device = torch.device("cpu")
    model = resnet20(num_classes=10).to(device)

    class_names = [f"class_{i}" for i in range(10)]
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    results_dir = temp_dirs["results_dir"]
    plot_training_curve(
        train_losses=[1.0, 0.9],
        test_accuracies=[30.0, 35.0],
        save_path=os.path.join(results_dir, "training_curve.png"),
    )
    plot_lr_schedule(
        learning_rates=[0.1, 0.05],
        save_path=os.path.join(results_dir, "lr_schedule.png"),
    )
    plot_feature_maps(
        model,
        test_loader,
        device,
        save_path=os.path.join(results_dir, "feature_maps.png"),
    )
    plot_confusion_matrix(
        model,
        test_loader,
        device,
        class_names,
        save_path=os.path.join(results_dir, "confusion_matrix.png"),
    )
    plot_misclassified_samples(
        model,
        test_loader,
        device,
        class_names,
        mean,
        std,
        save_path=os.path.join(results_dir, "misclassified_samples.png"),
    )

    for name in [
        "training_curve.png",
        "lr_schedule.png",
        "feature_maps.png",
        "confusion_matrix.png",
        "misclassified_samples.png",
    ]:
        assert os.path.isfile(os.path.join(results_dir, name))

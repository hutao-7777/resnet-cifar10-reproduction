# ResNet-CIFAR10-Reproduction

PyTorch reproduction of **Deep Residual Learning for Image Recognition** (He et al., CVPR 2016) on the CIFAR-10 and CIFAR-100 datasets.

---

## Table of Contents

- [Introduction](#introduction)
- [Results](#results)
- [Pretrained Models](#pretrained-models)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Key Implementations](#key-implementations)
- [Training Details](#training-details)
- [Visualization](#visualization)
- [References](#references)

---

## Introduction

This repository provides a clean, minimal, and fully runnable implementation of ResNet for CIFAR-10/100 classification. It is designed for reproducing the original ResNet paper and for comparing ResNet with a PlainNet (no residual connections) ablation.

Key features:

- Hand-written `BasicBlock` and `PlainBlock`.
- Hand-written `ResNet-20`, `ResNet-32`, `ResNet-44`, `ResNet-56`, `ResNet-110`, and `PlainNet-20`.
- Standard CIFAR-10/100 data augmentation plus optional Cutout.
- SGD + Momentum + Weight Decay + configurable LR schedule (Cosine Annealing or MultiStep).
- Optional label smoothing for the classification loss.
- TensorBoard logging, automatic best-checkpoint saving, and per-epoch history export.
- YAML-based configuration with full command-line override support.
- Visualization scripts for training curves, learning-rate schedule, feature maps, confusion matrices, and misclassified samples.
- Continuous integration via GitHub Actions.

---

## Results

The following table reports the settings used in this project. Run the training commands below and fill in the accuracies in place of the `--.--%` placeholders.

| Model        | Params | Epochs | Batch Size | Initial LR | CIFAR-10 Test Accuracy |
| ------------ | ------ | ------ | ---------- | ---------- | ---------------------- |
| ResNet-20    | 0.27 M | 200    | 128        | 0.1        | **--.--%**             |
| ResNet-32    | 0.46 M | 200    | 128        | 0.1        | **--.--%**             |
| ResNet-44    | 0.66 M | 200    | 128        | 0.1        | **--.--%**             |
| ResNet-56    | 0.86 M | 200    | 128        | 0.1        | **--.--%**             |
| ResNet-110   | 1.73 M | 200    | 128        | 0.1        | **--.--%**             |
| PlainNet-20  | 0.27 M | 200    | 128        | 0.1        | **--.--%**             |

> **Note:** `--.--%` is a placeholder. Train the corresponding model and replace it with the achieved test accuracy.

---

## Pretrained Models

No official pretrained checkpoints are provided in this repository yet. You can train your own models using the commands in the [Quick Start](#quick-start) section and upload the resulting `checkpoints/best.pth` files to a release or external storage.

| Model       | Dataset  | Download | Test Accuracy |
| ----------- | -------- | -------- | ------------- |
| ResNet-20   | CIFAR-10 | TBD      | --.--%        |
| ResNet-32   | CIFAR-10 | TBD      | --.--%        |
| PlainNet-20 | CIFAR-10 | TBD      | --.--%        |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/hutao-7777/resnet-cifar10-reproduction.git
cd resnet-cifar10-reproduction
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt`:

```text
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.5.0
tensorboard>=2.13.0
tqdm>=4.65.0
numpy>=1.23.0
scikit-learn>=1.2.0
seaborn>=0.12.0
pyyaml>=6.0

# Development / linting / testing
black>=23.0
isort>=5.12
flake8>=6.0
pytest>=7.0
```

### 3. Train a model

#### Command-line

```bash
# Train ResNet-20 (default)
python train.py --model resnet20 --epochs 200 --batch_size 128 --lr 0.1

# Train ResNet-32 with MultiStep LR and Cutout
python train.py --model resnet32 --epochs 200 --batch_size 128 --lr 0.1 \
    --lr_scheduler multistep --milestones 82 123 --gamma 0.1 \
    --use_cutout --cutout_length 16

# Train PlainNet-20 for ablation comparison
python train.py --model plainnet20 --epochs 200 --batch_size 128 --lr 0.1

# Train on CIFAR-100
python train.py --model resnet20 --dataset cifar100 --epochs 200 --batch_size 128 --lr 0.1
```

#### YAML configuration

```bash
python train.py --config configs/resnet20.yaml

# Override specific YAML values from the command line
python train.py --config configs/resnet20.yaml --epochs 10 --use_cutout
```

During training, the dataset will be automatically downloaded to `./data`, TensorBoard logs will be written to `./runs/{model}`, the best checkpoint will be saved to `./checkpoints/best.pth`, and the per-epoch training history will be saved to `./runs/{model}/history.npz` for visualization.

### 4. Evaluate a checkpoint

```bash
python eval.py --checkpoint checkpoints/best.pth --model resnet20 --dataset cifar10
```

### 5. Visualize

```bash
# After training both models, compare them and visualize the ResNet checkpoint
python visualize.py \
    --checkpoint checkpoints/best.pth \
    --model resnet20 \
    --dataset cifar10 \
    --resnet_history runs/resnet20/history.npz \
    --plain_history runs/plainnet20/history.npz
```

`visualize.py` generates the following figures under `results/`:

- `training_curve.png` — Train loss and test accuracy vs. epoch.
- `lr_schedule.png` — Learning rate vs. epoch.
- `resnet_vs_plainnet.png` — Accuracy comparison between ResNet-20 and PlainNet-20.
- `feature_maps.png` — First 6 feature maps from the first convolutional layer.
- `confusion_matrix.png` — Heatmap of the confusion matrix on the test set.
- `misclassified_samples.png` — 4x4 grid of misclassified samples with True/Pred labels.

### 6. TensorBoard

```bash
tensorboard --logdir runs
```

Then open `http://localhost:6006` in your browser.

---

## Project Structure

```text
resnet-cifar10-reproduction/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI
├── configs/
│   ├── resnet20.yaml           # YAML config for ResNet-20
│   ├── resnet32.yaml           # YAML config for ResNet-32
│   └── plainnet20.yaml         # YAML config for PlainNet-20
├── models/
│   ├── __init__.py
│   ├── basic_block.py          # BasicBlock + PlainBlock
│   └── resnet.py               # ResNet-20/32/44/56/110 + PlainNet-20
├── utils/
│   ├── __init__.py
│   ├── common.py               # Shared constants and model registry
│   ├── data_loader.py          # CIFAR-10/100 dataloaders with Cutout
│   ├── evaluator.py            # Accuracy and confusion matrix computation
│   ├── model_utils.py          # Model helper utilities
│   └── trainer.py              # Training loop with TensorBoard & checkpointing
├── checkpoints/                # Saved model checkpoints (created at runtime)
├── data/                       # CIFAR-10/100 dataset (downloaded automatically)
├── runs/                       # TensorBoard logs (created at runtime)
├── results/                    # Generated figures (created at runtime)
├── .pre-commit-config.yaml     # Pre-commit hooks
├── eval.py                     # Evaluation entry point
├── requirements.txt            # Python dependencies
├── train.py                    # Training entry point
├── visualize.py                # Visualization scripts
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

---

## Key Implementations

- **Residual Connection**: Implemented in `BasicBlock` as `out += identity` before the final ReLU.
- **Identity Mapping & Downsampling**: A `1x1` convolutional shortcut is used when stride != 1 or input/output channels mismatch.
- **Batch Normalization**: Applied before every ReLU activation.
- **Kaiming Initialization**: Conv layers are initialized with `kaiming_normal_`.
- **CIFAR-Specific Input Stem**: Uses a single `3x3` convolution (no `7x7` and no max-pooling) to preserve 32x32 resolution.
- **Optimizer & Scheduler**: SGD with momentum 0.9, weight decay 1e-4, and either `CosineAnnealingLR` or `MultiStepLR`.
- **Data Augmentation**: RandomCrop, RandomHorizontalFlip, and optional Cutout.
- **PlainNet Ablation**: Same depth and width as ResNet-20 but without shortcuts, demonstrating the benefit of residuals.

---

## Training Details

| Hyperparameter    | Value                                                         |
| ----------------- | ------------------------------------------------------------- |
| Optimizer         | SGD                                                           |
| Momentum          | 0.9                                                           |
| Weight Decay      | 1e-4                                                          |
| Initial LR        | 0.1                                                           |
| LR Schedule       | Cosine Annealing (`T_max = num_epochs`) or MultiStep [82,123] |
| LR Decay Factor   | 0.1 (MultiStep only)                                          |
| Loss Function     | CrossEntropyLoss (with optional label smoothing)              |
| Batch Size        | 128                                                           |
| Epochs            | 200                                                           |
| Data Augmentation | RandomCrop(32, padding=4) + RandomHorizontalFlip (+ Cutout)   |
| CIFAR-10 Norm     | mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)   |
| CIFAR-100 Norm    | mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)   |

---

## Visualization

After running `visualize.py`, the following figures are saved under `results/`:

1. `training_curve.png` — Train loss and test accuracy vs. epoch.
2. `lr_schedule.png` — Learning rate vs. epoch.
3. `resnet_vs_plainnet.png` — Accuracy comparison between ResNet-20 and PlainNet-20.
4. `feature_maps.png` — First 6 feature maps from the first convolutional layer.
5. `confusion_matrix.png` — Heatmap of the confusion matrix on the test set.
6. `misclassified_samples.png` — 4x4 grid of misclassified samples.

---

## References

- **Deep Residual Learning for Image Recognition**
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. CVPR 2016.
  [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

- **CIFAR-10 and CIFAR-100 Datasets**
  [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

- **Improved Regularization of Convolutional Neural Networks with Cutout**
  Terrance DeVries, Graham W. Taylor. arXiv:1708.04552.
  [arXiv:1708.04552](https://arxiv.org/abs/1708.04552)

---

## License

This project is released under the MIT License for educational and research purposes.

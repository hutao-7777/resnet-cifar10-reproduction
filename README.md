# ResNet-CIFAR10-Reproduction

PyTorch reproduction of **Deep Residual Learning for Image Recognition** (He et al., CVPR 2016) on the CIFAR-10 dataset.

> 中文：本项目使用 PyTorch 从零实现了 ResNet-20 / ResNet-32 以及用于对比实验的 PlainNet-20，并提供完整的训练、评估与可视化流程，适合论文复现和面试展示。

---

## Table of Contents

- [Introduction](#introduction)
- [Results](#results)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Key Implementations](#key-implementations)
- [Training Details](#training-details)
- [Visualization](#visualization)
- [References](#references)

---

## Introduction

This repository provides a clean, minimal, and fully runnable implementation of ResNet for CIFAR-10 classification. It is designed for:

- Reproducing the original ResNet paper on CIFAR-10.
- Comparing ResNet with a PlainNet (no residual connections) ablation.
- Interview demos and educational purposes.

Key features:

- Hand-written `BasicBlock` and `PlainBlock`.
- Hand-written `ResNet-20`, `ResNet-32`, and `PlainNet-20`.
- Standard CIFAR-10 data augmentation.
- SGD + Momentum + Weight Decay + Cosine Annealing LR.
- TensorBoard logging and automatic best-checkpoint saving.
- Visualization scripts for training curves, feature maps, and confusion matrices.

---

## Results

The following table reports the settings used in this project. You can fill in the actual accuracies after training.

| Model        | Params | Epochs | Batch Size | Initial LR | CIFAR-10 Test Accuracy |
| ------------ | ------ | ------ | ---------- | ---------- | ---------------------- |
| ResNet-20    | 0.27 M | 200    | 128        | 0.1        | **--.--%**             |
| ResNet-32    | 0.46 M | 200    | 128        | 0.1        | **--.--%**             |
| PlainNet-20  | 0.27 M | 200    | 128        | 0.1        | **--.--%**             |

> 中文说明：
> - ResNet-20 / ResNet-32 参数量分别约为 0.27 M 和 0.46 M。
> - PlainNet-20 与 ResNet-20 结构相同，但去掉了残差连接，通常准确率明显更低，可用于展示残差连接的重要性。
> - 论文中 ResNet-20 在 CIFAR-10 上的准确率约为 91.25%，ResNet-32 约为 92.49%（具体数值可能因随机种子和实现细节略有差异）。

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/resnet-cifar10-reproduction.git
cd resnet-cifar10-reproduction
```

### 2. Install dependencies

We recommend using a virtual environment (e.g., `venv` or `conda`):

```bash
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Train a model

```bash
# Train ResNet-20 (default)
python train.py --model resnet20 --epochs 200 --batch_size 128 --lr 0.1

# Train ResNet-32
python train.py --model resnet32 --epochs 200 --batch_size 128 --lr 0.1

# Train PlainNet-20 for ablation comparison
python train.py --model plainnet20 --epochs 200 --batch_size 128 --lr 0.1
```

> 中文：训练过程中会自动下载 CIFAR-10 数据集到 `./data`，TensorBoard 日志写入 `./runs`，最佳模型保存到 `./checkpoints/best.pth`。

### 4. Evaluate a checkpoint

```bash
python eval.py --checkpoint checkpoints/best.pth --model resnet20
```

### 5. Visualize

```bash
# After training two models, compare their histories
python visualize.py \
    --checkpoint checkpoints/best.pth \
    --model resnet20 \
    --resnet_history runs/resnet20/history.npz \
    --plain_history runs/plainnet20/history.npz
```

> 中文：`visualize.py` 默认会生成 `results/training_curve.png`、`results/resnet_vs_plainnet.png`、`results/feature_maps.png` 和 `results/confusion_matrix.png`。

### 6. TensorBoard

```bash
tensorboard --logdir runs
```

Then open `http://localhost:6006` in your browser.

---

## Project Structure

```text
resnet-cifar10-reproduction/
├── models/
│   ├── __init__.py
│   ├── basic_block.py          # BasicBlock + PlainBlock
│   └── resnet.py               # ResNet-20/32/44 + PlainNet-20
├── utils/
│   ├── __init__.py
│   ├── data_loader.py          # CIFAR-10 dataloaders with augmentation
│   ├── trainer.py              # Training loop with TensorBoard & checkpointing
│   └── evaluator.py            # Accuracy and confusion matrix computation
├── checkpoints/                # Saved model checkpoints (created at runtime)
├── data/                       # CIFAR-10 dataset (downloaded automatically)
├── runs/                       # TensorBoard logs (created at runtime)
├── results/                    # Generated figures (created at runtime)
├── train.py                    # Training entry point
├── eval.py                     # Evaluation entry point
├── visualize.py                # Visualization scripts
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

---

## Key Implementations

- **Residual Connection（残差连接）**: Implemented in `BasicBlock` as `out += identity` before the final ReLU.
- **Identity Mapping & Downsampling**: A `1x1` convolutional shortcut is used when stride ≠ 1 or input/output channels mismatch.
- **Batch Normalization（批归一化）**: Applied before every ReLU activation.
- **Kaiming Initialization（Kaiming 初始化）**: Conv layers are initialized with `kaiming_normal_`.
- **CIFAR-Specific Input Stem**: Uses a single `3x3` convolution (no `7x7` and no max-pooling) to preserve 32x32 resolution.
- **Optimizer & Scheduler**: SGD with momentum 0.9, weight decay 1e-4, and `CosineAnnealingLR`.
- **PlainNet Ablation**: Same depth and width as ResNet-20 but without shortcuts, demonstrating the benefit of residuals.

---

## Training Details

| Hyperparameter | Value                                       |
| -------------- | ------------------------------------------- |
| Optimizer      | SGD                                         |
| Momentum       | 0.9                                         |
| Weight Decay   | 1e-4                                        |
| Initial LR     | 0.1                                         |
| LR Schedule    | Cosine Annealing (`T_max = num_epochs`)     |
| Loss Function  | CrossEntropyLoss                            |
| Batch Size     | 128                                         |
| Epochs         | 200                                         |
| Data Augmentation | RandomCrop(32, padding=4) + RandomHorizontalFlip |
| Normalization  | mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616) |

> 中文：这些超参数与论文中的标准 CIFAR-10 设置基本一致。

---

## Visualization

After running `visualize.py`, the following figures are saved under `results/`:

1. `training_curve.png` — Train loss and test accuracy vs. epoch.
2. `resnet_vs_plainnet.png` — Accuracy comparison between ResNet-20 and PlainNet-20.
3. `feature_maps.png` — First 6 feature maps from the first convolutional layer.
4. `confusion_matrix.png` — Heatmap of the confusion matrix on the test set.

---

## References

- **Deep Residual Learning for Image Recognition**
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. CVPR 2016.
  [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

- **CIFAR-10 Dataset**
  [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## License

This project is released under the MIT License for educational and research purposes.

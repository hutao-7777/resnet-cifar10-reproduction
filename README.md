# ResNet-CIFAR10-Reproduction

PyTorch 复现 **Deep Residual Learning for Image Recognition**（He et al., CVPR 2016）在 CIFAR-10/100 数据集上的图像分类实验。项目以模块化、工程化的方式实现了 ResNet 家族与 PlainNet 消融网络，支持丰富的训练策略、完整的可视化与自动化 CI。

---

## 目录

- [项目简介](#项目简介)
- [已实现功能](#已实现功能)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [核心实现](#核心实现)
- [训练细节](#训练细节)
- [可视化说明](#可视化说明)
- [消融实验](#消融实验)
- [持续集成](#持续集成)
- [参考文献](#参考文献)
- [许可证](#许可证)

---

## 项目简介

本项目提供一个可直接运行、易于扩展的 ResNet CIFAR 复现代码库。核心目标有两个：

1. **复现 ResNet 在 CIFAR 数据集上的分类性能**，支持 ResNet-20/32/44/56/110 等多种深度。
2. **通过 PlainNet-20 消融实验验证残差连接的有效性**——PlainNet-20 与 ResNet-20 拥有完全相同的网络深度、宽度和卷积结构，唯一区别是移除了残差连接。

代码采用模块化设计，将模型定义、数据加载、训练循环、评估指标和可视化脚本分离，便于学习、调试和二次开发。

---

## 已实现功能

### 网络模型

- `BasicBlock` 与 `PlainBlock` 手写实现
- `ResNet-20/32/44/56/110` 五种深度
- `PlainNet-20` 消融网络（与 ResNet-20 同结构、无残差连接）
- Kaiming 初始化与 CIFAR 专用输入茎（3×3 conv，无 7×7 与 max-pool）

### 数据集

- 支持 `CIFAR-10` 与 `CIFAR-100`
- 自动下载到 `./data`
- 标准数据增强：RandomCrop、RandomHorizontalFlip
- 可选 Cutout 数据增强

### 训练策略

- 优化器：SGD + Momentum 0.9 + Weight Decay 1e-4
- 学习率调度：
  - `CosineAnnealingLR`
  - `MultiStepLR`（可自定义 milestones 与 gamma）
- 损失函数：`CrossEntropyLoss`，支持 Label Smoothing
- TensorBoard 日志记录
- 自动保存验证准确率最高的检查点
- 每轮训练历史导出为 `.npz`（train_losses / test_accuracies / learning_rates）

### 配置方式

- 命令行参数直接运行
- YAML 配置文件加载，命令行参数可覆盖 YAML 默认值
- 提供 `configs/resnet20.yaml`、`configs/resnet32.yaml`、`configs/plainnet20.yaml`

### 可视化

运行 `visualize.py` 可一键生成 6 类图表：

1. 训练曲线（train loss + test accuracy）
2. 学习率曲线
3. ResNet vs PlainNet 对比曲线
4. 第一层卷积特征图
5. 混淆矩阵热力图
6. 错误分类样本 4×4 网格

### 代码质量

- 符合 PEP8 规范
- 通过 `black`、`isort`、`flake8` 检查
- 提供合成数据冒烟测试
- GitHub Actions CI 自动化检查

---

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/hutao-7777/resnet-cifar10-reproduction.git
cd resnet-cifar10-reproduction
```

### 2. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

`requirements.txt` 内容如下：

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

### 3. 命令行训练

```bash
# ResNet-20 on CIFAR-10
python train.py --model resnet20 --epochs 200 --batch_size 128 --lr 0.1

# ResNet-32 + MultiStepLR + Cutout
python train.py --model resnet32 --epochs 200 --batch_size 128 --lr 0.1 \
    --lr_scheduler multistep --milestones 82 123 --gamma 0.1 \
    --use_cutout --cutout_length 16

# ResNet-20 on CIFAR-100
python train.py --model resnet20 --dataset cifar100 --epochs 200 --batch_size 128 --lr 0.1

# PlainNet-20 消融实验
python train.py --model plainnet20 --epochs 200 --batch_size 128 --lr 0.1
```

### 4. YAML 配置训练

```bash
python train.py --config configs/resnet20.yaml

# 命令行覆盖 YAML 参数
python train.py --config configs/resnet20.yaml --epochs 10 --use_cutout
```

训练过程中：

- CIFAR 数据集自动下载到 `./data`
- TensorBoard 日志写入 `./runs/{model}`
- 最优检查点保存到 `./checkpoints/best.pth`
- 训练历史保存到 `./runs/{model}/history.npz`

### 5. 评估

```bash
python eval.py --checkpoint checkpoints/best.pth --model resnet20 --dataset cifar10
```

### 6. 可视化

```bash
python visualize.py \
    --checkpoint checkpoints/best.pth \
    --model resnet20 \
    --dataset cifar10 \
    --resnet_history runs/resnet20/history.npz \
    --plain_history runs/plainnet20/history.npz
```

### 7. TensorBoard

```bash
tensorboard --logdir runs
```

浏览器访问 `http://localhost:6006`。

---

## 项目结构

```text
resnet-cifar10-reproduction/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI
├── configs/
│   ├── resnet20.yaml           # ResNet-20 默认配置
│   ├── resnet32.yaml           # ResNet-32 默认配置
│   └── plainnet20.yaml         # PlainNet-20 默认配置
├── models/
│   ├── __init__.py
│   ├── basic_block.py          # BasicBlock + PlainBlock
│   └── resnet.py               # ResNet-20/32/44/56/110 + PlainNet-20
├── utils/
│   ├── __init__.py
│   ├── common.py               # 共享常量与模型注册表
│   ├── data_loader.py          # CIFAR-10/100 数据加载 + Cutout
│   ├── evaluator.py            # 准确率、混淆矩阵
│   ├── model_utils.py          # count_parameters 等工具
│   └── trainer.py              # 训练循环、TensorBoard、检查点
├── tests/
│   └── test_smoke.py           # 合成数据冒烟测试
├── checkpoints/                # 保存的模型检查点（运行时生成）
├── data/                       # 自动下载的数据集（运行时生成）
├── results/                    # 可视化图片（运行时生成）
├── runs/                       # TensorBoard 日志（运行时生成）
├── .pre-commit-config.yaml     # pre-commit hooks
├── .gitignore
├── eval.py                     # 评估入口
├── requirements.txt            # Python 依赖
├── train.py                    # 训练入口
├── visualize.py                # 可视化入口
└── README.md                   # 本文件
```

---

## 核心实现

- **残差连接**：在 `BasicBlock` 中通过 `out += identity` 实现，位于最终 ReLU 之前。
- **下采样**：当 stride ≠ 1 或通道数不匹配时，使用 1×1 卷积投影 shortcut。
- **Batch Normalization**：每个卷积层后、ReLU 前加入 BN。
- **Kaiming 初始化**：卷积层使用 `kaiming_normal_`，BN 层初始化为 weight=1/bias=0。
- **CIFAR 输入茎**：使用 3×3 卷积，不使用 ImageNet 版的 7×7 卷积与 max-pooling，保留 32×32 分辨率。
- **学习率调度**：支持 `CosineAnnealingLR` 与 `MultiStepLR`，通过 `--lr_scheduler` 切换。
- **数据增强**：标准 RandomCrop + RandomHorizontalFlip，可选 Cutout。
- **损失函数**：`CrossEntropyLoss(label_smoothing=...)`，默认 label_smoothing=0.0。
- **训练历史**：每轮保存 `history.npz`，包含 `train_losses`、`test_accuracies`、`learning_rates`。

---

## 训练细节

| 超参数         | 默认值                                                       |
| -------------- | ------------------------------------------------------------ |
| 优化器         | SGD                                                          |
| Momentum       | 0.9                                                          |
| Weight Decay   | 1e-4                                                         |
| 初始学习率     | 0.1                                                          |
| 学习率调度     | CosineAnnealingLR（可选 MultiStepLR: [82, 123], gamma=0.1） |
| 损失函数       | CrossEntropyLoss（支持 Label Smoothing）                     |
| Batch Size     | 128                                                          |
| Epochs         | 200                                                          |
| 数据增强       | RandomCrop(32, padding=4) + RandomHorizontalFlip             |
| Cutout         | 可选，默认长度 16                                            |
| CIFAR-10 归一化 | mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616) |
| CIFAR-100 归一化| mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761) |

### 模型规模

| 模型        | 参数量  |
| ----------- | ------- |
| ResNet-20   | 0.27 M  |
| ResNet-32   | 0.46 M  |
| ResNet-44   | 0.66 M  |
| ResNet-56   | 0.86 M  |
| ResNet-110  | 1.73 M  |
| PlainNet-20 | 0.27 M  |

> 注：实验尚未跑完 200 epoch，最终测试准确率待补充。

---

## 可视化说明

运行 `visualize.py` 后，`results/` 目录下生成：

| 文件名                         | 说明                               |
| ------------------------------ | ---------------------------------- |
| `training_curve.png`           | 训练损失与测试准确率随 epoch 变化  |
| `lr_schedule.png`              | 学习率随 epoch 变化                |
| `resnet_vs_plainnet.png`       | ResNet-20 与 PlainNet-20 准确率对比 |
| `feature_maps.png`             | 第一层卷积输出的前 6 张特征图      |
| `confusion_matrix.png`         | 测试集混淆矩阵热力图               |
| `misclassified_samples.png`    | 4×4 错误分类样本网格，标注真实标签与预测标签 |

---

## 消融实验

本项目通过 `PlainNet-20` 与 `ResNet-20` 的对比，验证残差连接（residual connection）的有效性：

- **相同点**：两者均为 20 层网络，使用相同数量、相同通道数的 3×3 卷积层与 BatchNorm。
- **不同点**：`ResNet-20` 在每个 block 中加入 shortcut 连接，`PlainNet-20` 完全移除 shortcut。
- **预期结果**：在相同训练配置下，`ResNet-20` 的测试准确率显著高于 `PlainNet-20`，从而直观说明残差学习对深层网络训练的重要性。

训练并对比两者的命令：

```bash
python train.py --model resnet20 --epochs 200 --batch_size 128 --lr 0.1
python train.py --model plainnet20 --epochs 200 --batch_size 128 --lr 0.1
```

然后使用 `visualize.py` 的 `--resnet_history` 与 `--plain_history` 参数绘制对比曲线。

---

## 持续集成

本项目配置了 GitHub Actions CI（`.github/workflows/ci.yml`），在 Python 3.8/3.9/3.10 环境下执行：

- `black --check .`
- `isort --check-only --profile black .`
- `flake8 --max-line-length=100 ...`
- `pytest`
- 1 epoch 训练冒烟测试

同时提供 `.pre-commit-config.yaml`，可在本地提交前自动运行 `black`、`isort`、`flake8`。

---

## 参考文献

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). **Deep Residual Learning for Image Recognition**. CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- [CIFAR-10 and CIFAR-100 Datasets](https://www.cs.toronto.edu/~kriz/cifar.html)
- DeVries, T., & Taylor, G. W. (2017). **Improved Regularization of Convolutional Neural Networks with Cutout**. arXiv:1708.04552. [arXiv:1708.04552](https://arxiv.org/abs/1708.04552)

---

## 许可证

本项目基于 MIT 许可证发布，仅供学习和研究使用。

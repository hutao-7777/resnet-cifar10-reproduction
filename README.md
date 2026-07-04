# ResNet-CIFAR10-Reproduction

一个用 PyTorch 复现 ResNet 在 CIFAR-10 上图像分类的小项目。

## 实验结果

| 模型      | 参数量 | 训练轮数 | 批次大小 | 初始学习率 | CIFAR-10 测试准确率 |
| --------- | ------ | -------- | -------- | ---------- | ------------------- |
| ResNet-20 | 0.27 M | 10       | 128      | 0.1        | **81.65%**          |
| ResNet-32 | 0.46 M | -        | -        | -          | 未训练              |
| ResNet-44 | 0.66 M | -        | -        | -          | 未训练              |
| ResNet-56 | 0.86 M | -        | -        | -          | 未训练              |
| ResNet-110| 1.73 M | -        | -        | -          | 未训练              |
| PlainNet-20| 0.27 M| -        | -        | -          | 未训练              |

> 当前只完成了 ResNet-20 的 10 轮训练，最佳测试准确率为 81.65%。

## 环境准备

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

## 快速开始

### 训练

```bash
python train.py --model resnet20 --epochs 10 --batch_size 128 --lr 0.1
```

其他可选参数：

```bash
python train.py --model resnet20 --epochs 10 --lr_scheduler multistep --use_cutout
```

使用 YAML 配置：

```bash
python train.py --config configs/resnet20.yaml --epochs 10
```

训练过程中会自动下载 CIFAR-10 数据集到 `./data`，TensorBoard 日志保存到 `./runs/resnet20`，最佳模型保存到 `./checkpoints/best.pth`，训练历史保存到 `./runs/resnet20/history.npz`。

### 评估

```bash
python eval.py --checkpoint checkpoints/best.pth --model resnet20 --dataset cifar10
```

### 可视化

```bash
python visualize.py \
    --checkpoint checkpoints/best.pth \
    --model resnet20 \
    --dataset cifar10 \
    --resnet_history runs/resnet20/history.npz
```

可视化结果保存在 `./results/` 目录下：

- `training_curve.png`：训练损失与测试准确率曲线
- `lr_schedule.png`：学习率变化曲线
- `feature_maps.png`：第一层卷积特征图
- `confusion_matrix.png`：测试集混淆矩阵
- `misclassified_samples.png`：错误分类样本 4×4 网格

### TensorBoard

```bash
tensorboard --logdir runs
```

然后访问 `http://localhost:6006`。

## 项目结构

```text
resnet-cifar10-reproduction/
├── .github/workflows/ci.yml    # GitHub Actions 自动化
├── configs/                    # YAML 配置文件
├── models/                     # ResNet / PlainNet 模型定义
├── tests/                      # 冒烟测试
├── utils/                      # 数据加载、训练、评估工具
├── checkpoints/                # 训练保存的模型（运行时生成）
├── data/                       # CIFAR 数据集（运行时下载）
├── results/                    # 可视化图片（运行时生成）
├── runs/                       # TensorBoard 日志（运行时生成）
├── eval.py                     # 评估脚本
├── train.py                    # 训练脚本
├── visualize.py                # 可视化脚本
├── requirements.txt            # 依赖
└── README.md                   # 本文件
```

## 训练配置

本次 ResNet-20 训练使用的配置：

| 超参数       | 值                                    |
| ------------ | ------------------------------------- |
| 优化器       | SGD                                   |
| 动量         | 0.9                                   |
| 权重衰减     | 1e-4                                  |
| 初始学习率   | 0.1                                   |
| 学习率调度   | Cosine Annealing                      |
| 损失函数     | CrossEntropyLoss                      |
| 批次大小     | 128                                   |
| 训练轮数     | 10                                    |
| 数据增强     | RandomCrop + RandomHorizontalFlip     |
| 归一化       | mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616) |

## 参考

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

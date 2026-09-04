# CIARD IJCV 扩展版改进方案总结

> **历史 0830 设计说明。** 完整 0830 矩阵没有替代 0624 MobileNet-V2
> 或 0703 ResNet-18 主线，本文提出的 SARD/Label Smoothing/Adaptive
> Temperature 已退出两个顶层模型目录的活动实现。当前 0903 候选身份、
> 参数和未评测状态以仓库 `README.md` 及两个模型 README 为准；本文其余
> 内容只保留用于追溯当时的设计判断。

## 背景

CIARD (Cyclic Iterative Adversarial Robustness Distillation) 已被 ICCV 2025 接收。为了投稿 IJCV 期刊，需要对其进行扩展改进。本仓库包含两个独立的实验版本：

- `CIARD_Expansion_mobilenetv2_cifar10/` — MobileNet-V2 学生模型
- `CIARD_Expansion_resnet18_cifar10/` — ResNet-18 学生模型

## 当前实验结果

| 学生模型 | Clean Acc | 白盒鲁棒性 | 黑盒鲁棒性 | 状态 |
|----------|-----------|------------|------------|------|
| MobileNet-V2 | **高于 baseline** | **高于 baseline** | **高于 baseline** | 全指标正向提升 |
| ResNet-18 | 略低于 baseline | 略低于 baseline | **高于 baseline** | 白盒/干净有差距 |

## 根本原因分析

经过 10+ 轮的不同改进尝试（PCGrad 梯度手术、模型权重平均、FGSM anchor、自适应温度等），我们发现：

1. **MobileNet-V2**：其深度可分离卷积结构天然限制了 teacher-margin 梯度的反向传播，因此新增的 teacher-margin 能在不干扰白盒特征的情况下提升黑盒迁移鲁棒性。

2. **ResNet-18**：残差连接使得 teacher-margin 梯度能够直接流向浅层卷积层，干扰了决定白盒鲁棒性的特征。这导致了 Clean Acc 和白盒鲁棒性的系统性下降。

关键发现：问题不在于 teacher-margin 本身无效（它确实提升了黑盒鲁棒性），而在于 **teacher 在 x_adv 上的过度自信预测产生极端的 KL 散度梯度，学生被迫盲目模仿 teacher 的边界瑕疵**。

## 核心改进：Label Smoothing + Adaptive Temperature

### 1. Label Smoothing 正则化 teacher 标签

**问题**：Robust teacher 在对抗样本上给出接近 one-hot 的过度自信预测（即使错了也极度自信）。KL 散度在这种情况下梯度极大，导致学生被迫模仿 teacher 的每一个边界细节。

**解决方案**：对 teacher 的软标签应用 label smoothing：

```python
num_classes = robust_target.size(-1)
alpha = 0.1  # Label smoothing factor
robust_target = robust_target * (1 - alpha) + alpha / num_classes
```

这使得：
- 原本 100% 置信度的预测降到约 90%，同时将所有其他类从 0% 升到微小概率
- KL 散度中的 log(0) 消失，梯度变得平滑
- 学生不再被强制模仿 teacher 的微不足道的错误

### 2. Adaptive Temperature

**问题**：训练初期 teacher 在对抗样本上的预测质量很差（类似随机猜测），此时使用标准温度会导致大量噪声标签。

**解决方案**：从较高的温度开始（T=2.0），随训练进行线性衰减到 T=1.0：

```python
progress = min(1.0, epoch / 150)
temp_scale = 2.0 - 1.0 * progress  # 2.0 -> 1.0
temp_adv = max(1.0, temp_adv * temp_scale)
```

这使得：
- 训练初期：温度较高，teacher 的噪声标签被软化（减少误导）
- 训练后期：温度回到 1.0，teacher 的高质量标签发挥最大效果

## 配置参数

两个版本都新增了以下配置项（在 `CIARD.py` 中的 `CFG` 字典内）：

```python
"use_label_smoothing": True,      # 启用 Label Smoothing 正则化
"ls_alpha": 0.1,                  # Label Smoothing 强度 (0.1 = 标准值)
"use_adaptive_temp": True,        # 启用自适应温度
"temp_init_scale": 2.0,           # 初始温度倍数
"temp_decay_epochs": 150,         # 从初始温度衰减到 1.0 的 epoch 数
```

## 修改代码的具体位置

### MobileNet 版本
文件：`CIARD_Expansion_mobilenetv2_cifar10/CIARD.py`

**位置 1**（配置区，第 92 行后）：
- 添加 Label Smoothing 和 Adaptive Temperature 的配置项

**位置 2**（训练循环，第 479 行）：
- 在 `kl_Loss1/kl_Loss2` 计算后，插入 Label Smoothing 逻辑

**位置 3**（温度调整，第 550 行后）：
- 在温度裁剪后，插入 Adaptive Temperature 缩放

### ResNet 版本
文件：`CIARD_Expansion_resnet18_cifar10/CIARD.py`

同样的三个位置，应用相同的改动。

## 预期效果

| 指标 | MobileNet-V2 预期 | ResNet-18 预期 |
|------|-------------------|----------------|
| Clean Acc | 维持或小幅提升 | 提升 0.5-1.0pp（缩小与 baseline 的差距）|
| 白盒 FGSM | 维持提升 | 提升 0.5-1.0pp |
| 白盒 PGD | 维持提升 | 维持或小幅提升 |
| 黑盒指标 | 维持提升 | 稳定提升 |

## 为什么这些改进有效

| 改进 | 解决的问题 | 提升方向 |
|------|----------|---------|
| Label Smoothing | teacher 的过度自信边界 | Clean + White-box |
| Adaptive Temperature | 训练初期的 KL 不稳定 | Clean + White-box |
| 不动 teacher-margin | 黑盒迁移鲁棒性已证明有效 | 保持 Black-box 增益 |

## 运行实验

```bash
# MobileNet 版本
cd CIARD_Expansion_mobilenetv2_cifar10
python CIARD.py

# ResNet 版本
cd CIARD_Expansion_resnet18_cifar10
python CIARD.py
```

两个版本默认都包含 Label Smoothing + Adaptive Temperature。如需关闭某项改进，在 CIARD.py 的 CFG 中设置：

```python
"use_label_smoothing": False,
"use_adaptive_temp": False,
```

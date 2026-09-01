# CIFAR-10 / ResNet-18 当前综合主线

## 身份与来源

- 配置：0703 `r18_pcgrad_optuna_transfer`
- prefix：`Cifar10_ResNet18_0703_pcgrad_optuna_transfer`
- 整理前冻结副本：`my_temp_best/CIARD_Expansion_resnet18_cifar10_v1`
- 归档后来源：`CIARD_Expansion_before0824/my_temp_best/CIARD_Expansion_resnet18_cifar10_v1`
- 证据：`结果分析/CIARD_Expansion0703_pcgrad_resnet18_variants_结果分析.md`

该配置是当前 ResNet-18 综合主线。0712 的 `v1.2 previous_best_control` 基本复现了它；Label Smoothing/Adaptive Temperature 新组合没有形成可替代的全面收益。

## 与论文 CIARD baseline 对比

白盒基线取自 [CIARD 补充材料](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Lu_CIARD_Cyclic_Iterative_ICCV_2025_supplemental.pdf) Table 1 的 ResNet-18 / CIFAR-10 / CIARD 行，黑盒基线取自 [CIARD 主论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Lu_CIARD_Cyclic_Iterative_Adversarial_Robustness_Distillation_ICCV_2025_paper.pdf) Table 5 的对应行。所有数值单位均为百分比，括号内为当前主线相对论文 baseline 的百分点（pp）变化。

| 参数设置 | Clean | 白盒 FGSM | 白盒 PGDsat | 白盒 PGDtrades | 白盒 CW∞ | 黑盒 PGDtrades | 黑盒 Square | 黑盒 CW∞ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 论文 CIARD baseline | 88.87 | 61.88 | 51.70 | 54.46 | 50.61 | 66.28 | 80.03 | 64.79 |
| 当前主线（训练 seed 0） | 88.66 (-0.21 pp) | 61.33 (-0.55 pp) | 52.21 (+0.51 pp) | 54.76 (+0.30 pp) | 51.27 (+0.66 pp) | 66.60 (+0.32 pp) | 80.09 (+0.06 pp) | 65.02 (+0.23 pp) |

### 辅助均值比较

| 参数设置 | 白盒均值 | 黑盒均值 | 7 项鲁棒均值 | 8 项综合均值 |
| --- | ---: | ---: | ---: | ---: |
| 论文 CIARD baseline | 54.66 | 70.37 | 61.39 | 64.83 |
| 当前主线（训练 seed 0） | 54.89 (+0.23 pp) | 70.57 (+0.20 pp) | 61.61 (+0.22 pp) | 64.99 (+0.17 pp) |

白盒均值是 4 项白盒攻击的等权平均，黑盒均值是 3 项黑盒攻击的等权平均；7 项鲁棒均值汇总全部攻击，8 项综合均值再加入 Clean。均值及其差值均直接由各组成项计算，最后分别四舍五入到两位小数，因此个别“已显示均值相减”可能有 0.01 pp 的舍入差；它们是项目内部的辅助比较指标，不是论文的 W-R 指标。

当前主线的 Clean 和 FGSM 分别低于论文 baseline 0.21 pp 和 0.55 pp，其余六个单项均高于 baseline；7 项鲁棒均值和 8 项综合均值分别提高 0.22 pp 和 0.17 pp。结果来自训练 seed 0 的 `student_best.pth` 单次完整评测，微小差异不代表多次训练下的稳定收益。AutoAttack final robust 仍为 49.08，但不纳入本次八项论文对比或辅助均值。

可比性说明：本项目历史 evaluator 的 PGDtrades step size 为 `0.003`，而论文写明为 `2/255`，且历史随机起点 PGD、Square 等随机攻击未显式固定评测 seed。因此上述论文差值用于沿用项目历史口径的参考比较，不应表述为严格同协议复现。

## 使用说明

本目录只保存代码，不包含历史 checkpoint 或日志。`data`、`models` 指向项目根目录公共资源，输出目录已初始化。`train_4090.sbatch` 和 `eval_4090_best.sbatch` 的路径已经改到本目录；评测前必须先生成 `model/Cifar10_ResNet18_0703_pcgrad_optuna_transfer/student_best.pth`。

所有训练和评测作业只能由用户手动提交，Codex 不得执行 `sbatch`。

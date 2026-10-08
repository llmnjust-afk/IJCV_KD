# CIFAR-10 / ResNet-18 0906v2 G7 七项达标备份

## 身份与来源

- 配置：0906v2 G7 `resnet18_split_t025_n000_s120_w40_p081740`
- prefix：`Cifar10_ResNet18_0906v2_split_t025_n000_s120_w40_p081740`
- 参数：split_target_mix=True，参考 alpha=0.20，target alpha=0.25，nontarget alpha=0，start=120，warmup=40，push_lambda=0.081740；300 epochs、batch128、训练 seed 0。
- 备份来源：`run/0906v2/resnet18_split_t025_n000_s120_w40_p081740`
- 证据：`结果分析/0906v2_结果分析.md`；训练作业132043、评测作业132061，均为 COMPLETED / 0:0。

该配置是用户指定保留的七项超过论文 baseline 的候选，不覆盖原0703参考和0906v1 G3备份。非目标系数为0表示保留鲁棒教师的对抗预测条件分布，不是关闭非目标蒸馏。

## 与论文 CIARD baseline 对比

白盒基线取自 [CIARD 补充材料](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Lu_CIARD_Cyclic_Iterative_ICCV_2025_supplemental.pdf) Table 1 的 ResNet-18 / CIFAR-10 / CIARD 行，黑盒基线取自 [CIARD 主论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Lu_CIARD_Cyclic_Iterative_Adversarial_Robustness_Distillation_ICCV_2025_paper.pdf) Table 5 的对应行。所有数值单位均为百分比，括号内为G7相对论文 baseline 的百分点（pp）变化。

| 参数设置 | Clean | 白盒 FGSM | 白盒 PGDsat | 白盒 PGDtrades | 白盒 CW∞ | 黑盒 PGDtrades | 黑盒 Square | 黑盒 CW∞ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 论文 CIARD baseline | 88.87 | 61.88 | 51.70 | 54.46 | 50.61 | 66.28 | 80.03 | 64.79 |
| 0906v2 G7（训练 seed 0） | 88.91 (+0.04 pp) | 61.30 (-0.58 pp) | 51.91 (+0.21 pp) | 54.54 (+0.08 pp) | 51.53 (+0.92 pp) | 66.55 (+0.27 pp) | 80.20 (+0.17 pp) | 65.14 (+0.35 pp) |

### 辅助均值比较

| 参数设置 | 白盒均值 | 黑盒均值 | 7 项鲁棒均值 | 8 项综合均值 |
| --- | ---: | ---: | ---: | ---: |
| 论文 CIARD baseline | 54.66 | 70.37 | 61.39 | 64.83 |
| 0906v2 G7（训练 seed 0） | 54.82 (+0.16 pp) | 70.63 (+0.26 pp) | 61.60 (+0.20 pp) | 65.01 (+0.18 pp) |

白盒均值是4项白盒攻击的等权平均，黑盒均值是3项黑盒攻击的等权平均；7项鲁棒均值汇总全部攻击，8项综合均值再加入Clean。均值及差值从原始组成项分别计算后四舍五入，显示均值相减可能有0.01 pp的舍入差；它们是项目内部辅助指标，不是论文的W-R指标，也不是联合最坏情况鲁棒准确率。

**G7只有FGSM低于论文baseline 0.58 pp，其余七项均超过，但尚未实现全面提升。** 本批G1八项均值65.11%，高于G7的65.01%；G7相对G1仅Clean和Square提高，其余六项下降。结果来自seed 0的同一个EMA `student_best.pth`，微小差异不代表多次训练下的稳定收益。AutoAttack为49.25%，不纳入八项主表及辅助均值。沿用test-loader选模和未全部固定攻击seed的历史协议；PGDtrades步长为0.003，与论文2/255有差异。

## 使用说明

本目录只保存代码和说明，不包含checkpoint、日志、数据、教师权重或软链接。18个Python文件及requirements.txt与原G7逐字一致，两份Slurm脚本只适配本目录路径和作业名称。备份及已归档0906v1/0906v2的资源链接已移除，公共data/models保留；未来复跑需复制到新独立实验目录，准备资源、logs/slurm与model，并适配脚本路径和唯一prefix。

原评测权重路径：`/home/lixidong25/mycode/CIARD_Expansion/run/0906v2/resnet18_split_t025_n000_s120_w40_p081740/model/Cifar10_ResNet18_0906v2_split_t025_n000_s120_w40_p081740/student_best.pth`，SHA256为`93bf44010f6b7bd3919012e899d047ad35a9f74725acf2c3d8b28e774866719b`。同目录保留`eval_best_0906v2_132061.json`；原实验logs目录保留`train_stdout_132043.log`和`eval_best_stdout_132061.log`，完整源码哈希与清理追溯见本地版本台账。

本备份不能直接提交评测；评测前必须有对应运行的成功训练记录和非空student_best，完成检查要求variant、checkpoint路径/hash及训练源码hash与日志一致。所有训练和评测作业只能由用户手动提交，Codex不得执行sbatch。

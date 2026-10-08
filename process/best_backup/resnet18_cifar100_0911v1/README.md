# CIFAR-100 / ResNet-18 — 0911v1 C100-R1 已评测源码备份

2026-09-14：按用户指定，从 `run/0911v1/resnet18_cifar100_r5_v2_awp0p002_cr0p50` 保存完整轻量源码。**七项攻击指标全部高于论文，但Clean低2.73个百分点；本批七项攻击均值37.34%、八项均值40.55%，均高于R2。用户选定其作为当前CIFAR-100 ResNet最佳源码，不能将该选择写成对论文八项全面超越。**

对应的[活动入口](../../CIARD_Expansion_resnet18_cifar100/README.md)与本目录使用同一份已评测Python源码。

## 来源与固定配置

- 实验身份：`resnet18_cifar100_r5_v2_awp0p002_cr0p50`；prefix=`Cifar100_ResNet18_0909v1_r5_v2_awp0p002_cr0p50`，原身份及历史0909v1 prefix保持不变。
- CIFAR-100：50,000张训练 / 10,000张测试，100类，300 epochs、训练seed=0、全局batch=128，双卡满batch每卡64张。
- `training_views=2`、`awp_gamma=.002`、`consistency_weight=.50`、`method_start=120`、`method_warmup=40`、`consistency_temperature=.5`；第121轮启用双视图/新方法，第160轮达到完整强度。
- 保留R5的split KD：真实类alpha=.25、非目标类alpha=0、参考alpha=.20、start120/warmup40；非目标蒸馏仍参与训练。push=.081740、push_T=5、warmup80；clean CE=.036067、gate tau=.682852、start158/warmup116；teacher margin=.011316、start120/warmup89、tau=1.124788、cap=1.428932；teacher-margin PCGrad start120，EMA decay=.999642。
- 鲁棒教师WRN-70-16：`models/cifar100_linf_wrn70-16_without.pt`；自然教师WRN-22-6：`models/cifar100_wrn_22_6_finetuned_best.pth`，保留CIFAR-100模型内归一化和自然教师特征接口。

KD-AWP沿用自然/对抗KD代理梯度扰动卷积/线性权重；两视图各自执行原裁剪/翻转及PGD-10（预算8/255、步长2/255），完整原损失取均值，再加入 `lambda*r*mean(JS(softmax(z_adv1/.5),softmax(z_adv2/.5)))/100`。分母来自类别数，方法实现保持来源原样。

## 与论文 CIARD baseline 对比

单位为%；括号为本结果减论文的百分点差值。Clean和白盒取[补充材料 Table 1](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Lu_CIARD_Cyclic_Iterative_ICCV_2025_supplemental.pdf)的CIFAR-100列，黑盒取[主文 Table 5](https://openaccess.thecvf.com/content/ICCV2025/papers/Lu_CIARD_Cyclic_Iterative_Adversarial_Robustness_Distillation_ICCV_2025_paper.pdf)的CIFAR-100 / Robust列；本次依据已核验的本地论文与实验结果。

| 参数设置 | Clean | 白盒 FGSM | 白盒 PGDsat | 白盒 PGDtrades | 白盒 CW | 黑盒 PGDtrades | 黑盒 Square | 黑盒 CW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 论文 CIARD baseline | 65.73 | 34.47 | 28.05 | 29.45 | 24.43 | 42.29 | 49.76 | 41.44 |
| 0911v1 C100-R1（seed 0） | 63.00 (-2.73 pp) | 35.43 (+0.96 pp) | 30.90 (+2.85 pp) | 32.14 (+2.69 pp) | 27.65 (+3.22 pp) | 42.75 (+0.46 pp) | 50.54 (+0.78 pp) | 42.00 (+0.56 pp) |

### 辅助均值比较

| 参数设置 | 白盒均值 | 黑盒均值 | 7项攻击均值 | 8项综合均值 |
| --- | ---: | ---: | ---: | ---: |
| 论文 CIARD baseline | 29.10 | 44.50 | 35.70 | 39.45 |
| 0911v1 C100-R1 | 31.53 (+2.43 pp) | 45.10 (+0.60 pp) | 37.34 (+1.65 pp) | 40.55 (+1.10 pp) |

AutoAttack单独记录：**25.68%**。补充材料的AA表不是CIFAR-100基线，不套用其他数据集阈值。均值不含AA，分别为4项白盒、3项黑盒、7项攻击与含Clean的8项等权平均；均值和差值先计算再舍入，不代表联合最坏情况准确率。

## 完成证据

- 训练作业132781、评测作业133377均已核验 `COMPLETED / 0:0`；固定EMA best epoch=220，完整10,000张测试，未改选WA或拼接不同权重成绩。
- 原始实验：`/home/lixidong25/mycode/CIARD_Expansion/run/0911v1/resnet18_cifar100_r5_v2_awp0p002_cr0p50`。
- 原权重：`model/Cifar100_ResNet18_0909v1_r5_v2_awp0p002_cr0p50/student_best.pth`；该目录保留`training_complete.json`及`eval_best_0909v1_133377.json`。
- checkpoint SHA256：`5f75de9ac3bbb92c1a369c91c3a568aea846c965b3138ee55547b65fa8efa6a5`。
- evaluator SHA256：`b48f8fd80b48fbb633ce2e5fa9b98e6edd76b5efa7eccef65d849684e4fefc45`。
- 原始日志：`logs/train_stdout_132781.log`与`logs/eval_best_stdout_133377.log`，以及相应Slurm out/err；本地完整报告为`结果分析/0911v1_cifar100_结果分析.md`。
- 源码与同步文件哈希见[SYNC_MANIFEST.json](../../SYNC_MANIFEST.json)，论文对照汇总见[总README](../../README.md)。

## 使用与协议说明

本目录包含21份Python、原requirements和两份Slurm模板；Python/CFG/prefix/评测主体与来源逐字一致。脚本仅适配本目录工作/日志路径及作业名，保留rtx4090 / aias-compute-4，训练2×4090、评测1×4090、4CPU/16GB。

源码包不包含data/models资源链接、学生/教师权重、model/logs、训练完成记录或缓存，因此不是可直接提交的实验。已完成实验仍保留在原run目录，无需因本次备份重训或重评。未来复跑需复制到新的独立run目录，设置唯一prefix和匹配路径，准备公共资源及输出目录，所有GPU任务由用户手动提交。

依赖沿用已验证的ciard环境（Python3.8.20、torch1.10.0+cu113、torchvision0.11.1+cu113），CIFAR-100还使用该环境已安装的RobustBench，教师架构文件SHA256由run_checks.py锁定；原requirements保持原样，本次没有安装或升级依赖。

沿用test-loader的`(Clean+PGD proxy)/2`选择EMA best，存在测试集选择偏差；训练仅seed0，历史随机攻击未全部显式固定seed。PGDtrades评测20步、步长.003继承官方CIARD实现，与论文文字2/255不同；训练PGD-10仍为2/255。单次结果与论文数值的比较不等于严格同参数复现或多seed稳定性证明。

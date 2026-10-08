# CIARD 拓展论文讨论代码

2026-10-08 整理。本目录集中保存当前选定的两种学生模型、两个数据集的四份已测源码，供论文写作和方法讨论。先阅读 [CIARD扩展方法改进说明](CIARD扩展方法改进说明.md)，再按下表进入对应代码。

四份共享 CIARD 双教师与循环更新主干，以及可靠 push、门控 Clean CE、教师 margin、KD-AWP、双视图对抗一致性和学生 EMA。两个 ResNet 另用拆分蒸馏与 margin 梯度冲突投影；只有 CIFAR-100 ResNet H2 增加自然教师二元监督。因此采用“共同框架与明确配置差异”的表述。

## 当前四份源码

| 入口 | 已选实验 | 报告权重 | 八项超过论文 |
|---|---|---|---:|
| [MobileNet-V2 / CIFAR-10](CIARD_Expansion_mobilenetv2_cifar10/README.md) | 0917v1 M1 | EMA best，第250轮 | 8/8 |
| [ResNet-18 / CIFAR-10](CIARD_Expansion_resnet18_cifar10/README.md) | 0917v1 R4 | EMA best，第252轮 | 7/8 |
| [MobileNet-V2 / CIFAR-100](CIARD_Expansion_mobilenetv2_cifar100/README.md) | 0914v1 C100-M1 | EMA best，第230轮 | 8/8 |
| [ResNet-18 / CIFAR-100](CIARD_Expansion_resnet18_cifar100/README.md) | 1004-cifar100-binary-v1 C100-H2 | 固定190轮 EMA | 8/8 |

这里的“当前”指已选定的论文讨论主线，不代表每项都超过所有历史版本。CIFAR-10 ResNet R4 的 FGSM 为61.80%，比论文61.88%低0.08个百分点，其余七项及AA高于论文；正文和表格保留该差值。CIFAR-10 MobileNet M4仍在[M4复现参考](CIARD_Expansion_mobilenetv2_cifar10/M4_REPRODUCTION.md)中保留，不新增第五个默认入口。

## 目录与来源

- [CIARD](CIARD/README.md)：ICCV原始实现；[原论文](CIARD/Lu_CIARD_Cyclic_Iterative_Adversarial_Robustness_Distillation_ICCV_2025_paper.pdf)及[补充材料](CIARD/ICCV_CIARD_Supplementary.pdf)供对照。
- [process](process/README.md)：此前的开发、版本说明与历史备份；本次完整保留。
- 四个平级 `CIARD_Expansion_*` 目录：从 `process` 的同名入口复制。计算源码、配置、依赖和历史脚本不作清理或重构；新副本只调整说明和链接。
- [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)：复制来源、固定CFG、逐文件SHA256、原实验身份、权重及结果证据。
- [本次整理核验](../preparation_20261008/verification.json)：文件一致性、静态检查、结果与链接核验；不代表本次进行了新训练或测试。

## 阅读副本与运行入口

**四份目录用于阅读和追溯现有成绩，不能直接提交作业。** 未复制数据、公共教师、学生权重、日志或资源软链接。前三组脚本保留 `origin_code/0917v1-cifar10` 的工作与日志路径，H2脚本保留 `origin_code/1004` 的路径；H2还绑定原实验身份、父manifest和固定权重路径。脚本与校验程序一起作为历史源码保留，不能仅改一个目录就视为新的复现包。

未来复跑须另建独立run，重新准备身份、公共资源链接、输出目录及配套校验；所有GPU作业由用户手动提交。本次仅本地整理，不涉及新训练、评测或Git发布。

## 结果口径

单位为百分比，差值为本版本减对应论文CIARD值的百分点。Clean和白盒取补充材料Table 1（ResNet）或Table 2（MobileNet）的对应数据集列；黑盒取主文Table 5或Table 6的Robust列。CIFAR-10 AA取补充Table 5，CIFAR-100没有对应论文AA阈值。七项攻击均值和八项均值均不包含AA，也不代表联合最坏情况准确率；由原始正确数计算，差值计算后再四舍五入至两位。

前三组历史训练用test-loader的 `(Clean + PGD proxy)/2` 选择EMA best；H2训练不读取test选模，固定第190轮最后更新后的EMA，但其方法和终点参考过历史测试结果。均为已完成的单seed结果，微小差值不能推导统计稳定性。

保留官方历史评测实现：L∞预算8/255，PGDsat20步/2/255，PGDtrades20步/.003，CW30步/2/255，Square100 queries，以及AutoAttack。PGDtrades的.003与论文文字2/255有差异，来自继承的实现；训练仍为PGD10/2/255。前三组随机攻击未全部显式定seed，H2各指标攻击seed为0。下表是该冻结口径下与论文报告值的对照。

## MobileNet-V2 / CIFAR-10

来源 **0917v1 M1**；训练／评测作业 **134636／135053** 已归档为 COMPLETED / 0:0。下表全部指标来自EMA best，第250轮的同一权重、完整10000张测试。

| 指标 | 原论文 CIARD | 本版本 | 差值 pp |
|---|---:|---:|---:|
| Clean | 89.51 | 89.61 | +0.10 |
| 白盒 FGSM | 59.10 | 60.87 | +1.77 |
| 白盒 PGDsat | 47.67 | 50.67 | +3.00 |
| 白盒 PGDtrades | 50.71 | 53.36 | +2.65 |
| 白盒 CW | 46.88 | 49.01 | +2.13 |
| 黑盒 PGDtrades | 66.66 | 67.67 | +1.01 |
| Square 查询攻击 | 80.01 | 81.27 | +1.26 |
| 黑盒 CW | 66.12 | 66.14 | +0.02 |
| AutoAttack 单列 | 46.31 | 47.01 | +0.70 |
| 七项攻击均值 | 59.59 | 61.28 | +1.69 |
| 八项均值 | 63.33 | 64.83 | +1.49 |

[完整结果报告](../../../结果分析/0917v1_结果分析.md) · [评测日志](../../../run/0917v1/mobilenetv2_cifar10_v2_awp0p001_cr0p50/logs/eval_best_stdout_135053.log) · [原始结果JSON](../../../run/0917v1/mobilenetv2_cifar10_v2_awp0p001_cr0p50/model/Cifar10_MobileNetV2_0917v1_v2_awp0p001_cr0p50/eval_best_0909v1_135053.json) · [训练日志](../../../run/0917v1/mobilenetv2_cifar10_v2_awp0p001_cr0p50/logs/train_stdout_134636.log)

## ResNet-18 / CIFAR-10

来源 **0917v1 R4**；训练／评测作业 **134643／134905** 已归档为 COMPLETED / 0:0。下表全部指标来自EMA best，第252轮的同一权重、完整10000张测试。

| 指标 | 原论文 CIARD | 本版本 | 差值 pp |
|---|---:|---:|---:|
| Clean | 88.87 | 89.10 | +0.23 |
| 白盒 FGSM | 61.88 | 61.80 | -0.08 |
| 白盒 PGDsat | 51.70 | 52.06 | +0.36 |
| 白盒 PGDtrades | 54.46 | 54.86 | +0.40 |
| 白盒 CW | 50.61 | 51.24 | +0.63 |
| 黑盒 PGDtrades | 66.28 | 67.28 | +1.00 |
| Square 查询攻击 | 80.03 | 80.77 | +0.74 |
| 黑盒 CW | 64.79 | 65.62 | +0.83 |
| AutoAttack 单列 | 48.88 | 49.10 | +0.22 |
| 七项攻击均值 | 61.39 | 61.95 | +0.55 |
| 八项均值 | 64.83 | 65.34 | +0.51 |

[完整结果报告](../../../结果分析/0917v1_结果分析.md) · [评测日志](../../../run/0917v1/resnet18_cifar10_v2_awp0p002_cr0p75/logs/eval_best_stdout_134905.log) · [原始结果JSON](../../../run/0917v1/resnet18_cifar10_v2_awp0p002_cr0p75/model/Cifar10_ResNet18_0917v1_v2_awp0p002_cr0p75/eval_best_0909v1_134905.json) · [训练日志](../../../run/0917v1/resnet18_cifar10_v2_awp0p002_cr0p75/logs/train_stdout_134643.log)

## MobileNet-V2 / CIFAR-100

来源 **0914v1 C100-M1**；训练／评测作业 **133792／134582** 已归档为 COMPLETED / 0:0。下表全部指标来自EMA best，第230轮的同一权重、完整10000张测试。

| 指标 | 原论文 CIARD | 本版本 | 差值 pp |
|---|---:|---:|---:|
| Clean | 66.72 | 66.91 | +0.19 |
| 白盒 FGSM | 33.56 | 34.57 | +1.01 |
| 白盒 PGDsat | 27.02 | 28.61 | +1.59 |
| 白盒 PGDtrades | 28.95 | 30.06 | +1.11 |
| 白盒 CW | 25.54 | 27.28 | +1.74 |
| 黑盒 PGDtrades | 42.70 | 44.82 | +2.12 |
| Square 查询攻击 | 50.85 | 53.29 | +2.44 |
| 黑盒 CW | 42.85 | 43.67 | +0.82 |
| AutoAttack 单列 | — | 24.95 | — |
| 七项攻击均值 | 35.92 | 37.47 | +1.55 |
| 八项均值 | 39.77 | 41.15 | +1.38 |

[完整结果报告](../../../结果分析/0914v1_cifar100_结果分析.md) · [评测日志](../../../run/0914v1/mobilenetv2_cifar100_natorig_awp0p002_cr0p50/logs/eval_best_stdout_134582.log) · [原始结果JSON](../../../run/0914v1/mobilenetv2_cifar100_natorig_awp0p002_cr0p50/model/Cifar100_MobileNetV2_0914v1_natorig_awp0p002_cr0p50/eval_best_0909v1_134582.json) · [训练日志](../../../run/0914v1/mobilenetv2_cifar100_natorig_awp0p002_cr0p50/logs/train_stdout_133792.log)

## ResNet-18 / CIFAR-100

来源 **1004-cifar100-binary-v1 C100-H2**；训练／评测作业 **137534／137729** 已归档为 COMPLETED / 0:0。下表全部指标来自固定190轮 EMA的同一权重、完整10000张测试。

| 指标 | 原论文 CIARD | 本版本 | 差值 pp |
|---|---:|---:|---:|
| Clean | 65.73 | 65.96 | +0.23 |
| 白盒 FGSM | 34.47 | 34.52 | +0.05 |
| 白盒 PGDsat | 28.05 | 29.76 | +1.71 |
| 白盒 PGDtrades | 29.45 | 30.78 | +1.33 |
| 白盒 CW | 24.43 | 26.66 | +2.23 |
| 黑盒 PGDtrades | 42.29 | 43.52 | +1.23 |
| Square 查询攻击 | 49.76 | 51.90 | +2.14 |
| 黑盒 CW | 41.44 | 42.29 | +0.85 |
| AutoAttack 单列 | — | 24.22 | — |
| 七项攻击均值 | 35.70 | 37.06 | +1.36 |
| 八项均值 | 39.45 | 40.67 | +1.22 |

[完整结果报告](../../../结果分析/1004-cifar100-binary-v1_结果分析.md) · [评测日志](../../../run/1004-cifar100-binary-v1/resnet18_cifar100_nb_l2_w40_s0/logs/eval_stdout_137729.log) · [原始结果JSON](../../../run/1004-cifar100-binary-v1/resnet18_cifar100_nb_l2_w40_s0/model/Cifar100_ResNet18_1004_binary_v1_H2/eval_137729.json) · [训练日志](../../../run/1004-cifar100-binary-v1/resnet18_cifar100_nb_l2_w40_s0/logs/train_stdout_137534.log)

# CIFAR-10 / ResNet-18 — 0917v1 R4 已评测活动入口

**2026-10-08 论文讨论源码副本。** 本目录从 `../process/CIARD_Expansion_resnet18_cifar10` 完整复制，计算源码、固定配置、依赖与历史脚本保持原样；初次整理只调整说明和链接，现补充所选成功训练与评测日志。下文早于本日期的同步与核验记录均描述原版本。

[返回1008总导航](../README.md) · [中文方法改进说明](../CIARD扩展方法改进说明.md) · [本次来源清单](../SOURCE_MANIFEST.json)

**本目录用于阅读与追溯现有结果，不能直接提交。** 已附所选成功训练与评测日志，未打包数据、教师和学生权重或资源链接。下文 `run/`、`model/`、结果JSON、完整报告及校验记录仍在原实验或项目目录，仅供本地追溯。复跑须另建独立实验并重建身份、路径及校验绑定。

`logs/`保留本组训练与评测的stdout及对应Slurm `.out/.err`，共6份；stdout与对应`.out`逐字节一致，`.err`为空，未复制锁文件。四组共24份、约7.9 MB的归档范围见[总README](../README.md)，来源及hash见[来源清单](../SOURCE_MANIFEST.json)。

2026-09-19：按用户选择同步已测来源。同版冻结备份见[best_backup/resnet18_cifar10_0917v1_r4](../process/best_backup/resnet18_cifar10_0917v1_r4/README.md)。

R4按用户选择成为当前入口，八项中7项严格高于论文；FGSM=61.80%，仍低.08pp，严格超过61.88%需达到61.89%。AA=49.10%（+.22pp），八项均值65.34%。相对历史R5，FGSM只提高.04pp，八项均值及AA下降；这次源码选择不表示R4全面优于R5，也不表示ResNet已达成八项全面提升。

## 固定配置与方法

AWP gamma=0.002，consistency_weight=0.75。VARIANT_NAME=`resnet18_cifar10_v2_awp0p002_cr0p75`，prefix=`Cifar10_ResNet18_0917v1_v2_awp0p002_cr0p75`。

ResNet保留split KD target=.25/non-target=0/reference=.20、push=.081740、clean CE=.036067、teacher margin=.011316、PCGrad和EMA=.999642。

两组模型均沿用已测KD-AWP＋对抗双视图一致性方法，50,000张训练/10,000张测试，300 epochs、batch=128、seed=0、training_views=2、method_start=120、method_warmup=40、consistency_temperature=.5。实际CFG由原CIARD.py打印，完整配置及Python哈希见同步清单。同步没有改变损失、模型、教师、攻击、选模或评测输出。

教师保持raw输入的WRN-34-10（models/model_cifar_wrn.pt）与ResNet-56（models/nat_teacher_checkpoint/cifar10_resnnet56.pth），不增加外部Normalize。训练PGD-10、预算8/255、步长2/255；完整评测保留PGDsat20步/2/255、PGDtrades20步/.003、CW30步/2/255、Square100queries与AA。黑盒PGDtrades/CW由鲁棒教师生成迁移样本，Square查询学生。

沿用test-loader的(Clean+PGD proxy)/2选EMA best，存在测试集选择偏差；历史随机攻击未全部显式固定seed。PGDtrades步长.003继承官方实现，与论文文字2/255不同。本次是单seed固定checkpoint结果，不能据此宣称稳定的多seed优势或严格同协议复现；均值不代表联合最坏情况准确率。

## 已完成测试结果

单位为%，括号为本结果减同模型论文baseline的百分点差值。论文来源、历史对照及两个数据集结果见[历史结果总README](../process/README.md)。均值不含AA，先计算再舍入。

| Metric | Paper CIARD baseline | 0909v1 R5 (historical) (Δ vs paper) | 0917v1 R4 (Δ vs paper) |
| --- | ---: | ---: | ---: |
| Clean | 88.87 | 89.07 (+0.20 pp) | 89.10 (+0.23 pp) |
| White-box FGSM | 61.88 | 61.76 (-0.12 pp) | 61.80 (-0.08 pp) |
| White-box PGDsat | 51.70 | 52.34 (+0.64 pp) | 52.06 (+0.36 pp) |
| White-box PGDtrades | 54.46 | 55.01 (+0.55 pp) | 54.86 (+0.40 pp) |
| White-box CW | 50.61 | 51.77 (+1.16 pp) | 51.24 (+0.63 pp) |
| Black-box PGDtrades | 66.28 | 67.16 (+0.88 pp) | 67.28 (+1.00 pp) |
| Square (query-based) | 80.03 | 80.87 (+0.84 pp) | 80.77 (+0.74 pp) |
| Black-box CW | 64.79 | 65.71 (+0.92 pp) | 65.62 (+0.83 pp) |
| AutoAttack (separate) | 48.88 | 49.35 (+0.47 pp) | 49.10 (+0.22 pp) |
| Seven-attack mean | 61.39 | 62.09 (+0.70 pp) | 61.95 (+0.55 pp) |
| Eight-metric mean | 64.83 | 65.46 (+0.63 pp) | 65.34 (+0.51 pp) |

## 完成证据

- 已测来源：`/home/lixidong25/mycode/CIARD_Expansion/run/0917v1/resnet18_cifar10_v2_awp0p002_cr0p75`。
- 训练/评测：`134643 / 134905`，均为`COMPLETED / 0:0`；固定EMA best epoch=252，完整10000张测试集。
- 权重：`model/Cifar10_ResNet18_0917v1_v2_awp0p002_cr0p75/student_best.pth`；checkpoint SHA256：`208cc7dbe1a4158c8c6d3a41f0f442e3952dd92cf011f3419e00edb9845694e8`。
- evaluator SHA256：`86803bff2df8b0a88b7a4f43d82ca412f80ad42c7ec297cd8d97ceddc59b6bd3`。
- 结果：`run/0917v1/resnet18_cifar10_v2_awp0p002_cr0p75/model/Cifar10_ResNet18_0917v1_v2_awp0p002_cr0p75/eval_best_0909v1_134905.json`；历史JSON文件名保留0909v1，不代表来源批次。
- 仓库内日志：[训练](logs/train_stdout_134643.log)、[评测](logs/eval_best_stdout_134905.log)；对应Slurm输出保存在[logs/slurm](logs/slurm/)。
- 已核验CPU严格加载、完成标记、源码/权重/评测器哈希、九项正确数及日志/JSON一致性；原始权重和日志保留在原run。

## 源码模板与复跑

本目录包含20份Python，与已测run逐字一致，完整CFG和文件哈希见[SYNC_MANIFEST.json](../process/SYNC_MANIFEST.json)。训练与完整评测入口为CIARD.py、attack_eval.py，配套train_4090.sbatch与eval_4090_best.sbatch；其他辅助脚本保留历史用途。脚本在历史同步时修改了工作目录及日志路径；1008逐字保留该模板，工作和日志路径仍指向 `origin_code/0917v1-cifar10`，不能直接执行；保留原rtx4090分区、aias-compute-4、单4090、4CPU/16GB配置。

本包包含源码、依赖、说明和上述成功日志，不包含data/models链接、checkpoint、model目录或缓存，当前不能直接提交。未来复跑需复制到新的独立run，设置唯一身份/prefix、匹配评测路径和脚本目录，建立公共data/models链接及空输出目录。原已测run保持冻结，全部GPU训练与评测由用户手动提交；本次同步无需重训。

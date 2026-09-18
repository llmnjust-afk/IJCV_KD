# CIFAR-10 / MobileNet-V2 — 0917v1 M1 已评测活动入口

2026-09-19：按用户选择同步已测来源。同版冻结备份见[best_backup/mobilenetv2_cifar10_0917v1_m1](../best_backup/mobilenetv2_cifar10_0917v1_m1/README.md)。

M1与M4各自同一个固定checkpoint的八项均严格高于论文，AA分别47.01%与46.70%；两者黑盒CW均66.14%，仅高论文.02pp。按用户选择以M1作为默认源码：M1的FGSM、PGDtrades、白盒CW和AA更高；M4的Clean、PGDsat、黑盒PGDtrades和Square更高，八项均值64.89%高于M1的64.83%（M1精确64.825%）。M4−M1九项依次为+.41/−.16/+.08/−.02/−.25/+.31/+.14/0/−.31pp。两者均未全面超过历史0909v1 M2，M1/M4本次过线不等于稳定全胜。

M4配置和身份修改完整列在[M4复现说明](M4_REPRODUCTION.md)，其中包含M4自己的权重/评测来源，不混用M1权重。

## 固定配置与方法

AWP gamma=0.001，consistency_weight=0.5。VARIANT_NAME=`mobilenetv2_cifar10_v2_awp0p001_cr0p50`，prefix=`Cifar10_MobileNetV2_0917v1_v2_awp0p001_cr0p50`。

MobileNet保留push=.05、clean CE=.05、teacher margin=.01、EMA=.999和普通反传。

两组模型均沿用已测KD-AWP＋对抗双视图一致性方法，50,000张训练/10,000张测试，300 epochs、batch=128、seed=0、training_views=2、method_start=120、method_warmup=40、consistency_temperature=.5。实际CFG由原CIARD.py打印，完整配置及Python哈希见同步清单。同步没有改变损失、模型、教师、攻击、选模或评测输出。

教师保持raw输入的WRN-34-10（models/model_cifar_wrn.pt）与ResNet-56（models/nat_teacher_checkpoint/cifar10_resnnet56.pth），不增加外部Normalize。训练PGD-10、预算8/255、步长2/255；完整评测保留PGDsat20步/2/255、PGDtrades20步/.003、CW30步/2/255、Square100queries与AA。黑盒PGDtrades/CW由鲁棒教师生成迁移样本，Square查询学生。

沿用test-loader的(Clean+PGD proxy)/2选EMA best，存在测试集选择偏差；历史随机攻击未全部显式固定seed。PGDtrades步长.003继承官方实现，与论文文字2/255不同。本次是单seed固定checkpoint结果，不能据此宣称稳定的多seed优势或严格同协议复现；均值不代表联合最坏情况准确率。

## 已完成测试结果

单位为%，括号为本结果减同模型论文baseline的百分点差值。论文来源、历史对照及两个数据集结果见[总README](../README.md)。均值不含AA，先计算再舍入。

| Metric | Paper CIARD baseline | 0909v1 M2 (historical) (Δ vs paper) | 0917v1 M1 (Δ vs paper) | 0917v1 M4 (Δ vs paper) |
| --- | ---: | ---: | ---: | ---: |
| Clean | 89.51 | 89.79 (+0.28 pp) | 89.61 (+0.10 pp) | 90.02 (+0.51 pp) |
| White-box FGSM | 59.10 | 61.03 (+1.93 pp) | 60.87 (+1.77 pp) | 60.71 (+1.61 pp) |
| White-box PGDsat | 47.67 | 51.06 (+3.39 pp) | 50.67 (+3.00 pp) | 50.75 (+3.08 pp) |
| White-box PGDtrades | 50.71 | 53.53 (+2.82 pp) | 53.36 (+2.65 pp) | 53.34 (+2.63 pp) |
| White-box CW | 46.88 | 49.14 (+2.26 pp) | 49.01 (+2.13 pp) | 48.76 (+1.88 pp) |
| Black-box PGDtrades | 66.66 | 67.71 (+1.05 pp) | 67.67 (+1.01 pp) | 67.98 (+1.32 pp) |
| Square (query-based) | 80.01 | 81.36 (+1.35 pp) | 81.27 (+1.26 pp) | 81.41 (+1.40 pp) |
| Black-box CW | 66.12 | 65.95 (-0.17 pp) | 66.14 (+0.02 pp) | 66.14 (+0.02 pp) |
| AutoAttack (separate) | 46.31 | 47.15 (+0.84 pp) | 47.01 (+0.70 pp) | 46.70 (+0.39 pp) |
| Seven-attack mean | 59.59 | 61.40 (+1.80 pp) | 61.28 (+1.69 pp) | 61.30 (+1.71 pp) |
| Eight-metric mean | 63.33 | 64.95 (+1.61 pp) | 64.83 (+1.49 pp) | 64.89 (+1.56 pp) |

## 完成证据

- 已测来源：`/home/lixidong25/mycode/CIARD_Expansion/run/0917v1/mobilenetv2_cifar10_v2_awp0p001_cr0p50`。
- 训练/评测：`134636 / 135053`，均为`COMPLETED / 0:0`；固定EMA best epoch=250，完整10000张测试集。
- 权重：`model/Cifar10_MobileNetV2_0917v1_v2_awp0p001_cr0p50/student_best.pth`；checkpoint SHA256：`3956b121747007143f5a9ddd561dafe080b197a4670f95d20ee831551dc8bc7e`。
- evaluator SHA256：`ea395e24cdb7f1089a4e10aca7c3911d09e5ec55cac8a8d48c702a34dfeac152`。
- 结果：`run/0917v1/mobilenetv2_cifar10_v2_awp0p001_cr0p50/model/Cifar10_MobileNetV2_0917v1_v2_awp0p001_cr0p50/eval_best_0909v1_135053.json`；历史JSON文件名保留0909v1，不代表来源批次。
- 日志：`run/0917v1/mobilenetv2_cifar10_v2_awp0p001_cr0p50/logs/train_stdout_134636.log`、`run/0917v1/mobilenetv2_cifar10_v2_awp0p001_cr0p50/logs/eval_best_stdout_135053.log`。
- 已核验CPU严格加载、完成标记、源码/权重/评测器哈希、九项正确数及日志/JSON一致性；原始权重和日志保留在原run。

## 源码模板与复跑

本目录包含23份Python，与已测run逐字一致，完整CFG和文件哈希见[SYNC_MANIFEST.json](../SYNC_MANIFEST.json)。训练与完整评测入口为CIARD.py、attack_eval.py，配套train_4090.sbatch与eval_4090_best.sbatch；其他辅助脚本保留历史用途。脚本只修改工作目录及日志路径；保留原rtx4090分区、aias-compute-2、单4090、4CPU/16GB配置。

本包只有源码、依赖和说明，不包含data/models链接、checkpoint、model/logs或缓存，当前不能直接提交。未来复跑需复制到新的独立run，设置唯一身份/prefix、匹配评测路径和脚本目录，建立公共data/models链接及空输出目录。原已测run保持冻结，全部GPU训练与评测由用户手动提交；本次同步无需重训。

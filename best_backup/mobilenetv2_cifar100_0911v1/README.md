# CIFAR-100 / MobileNet-V2 — 0911v1 C100-M1 已评测源码备份

2026-09-14：按用户指定，从 `run/0911v1/mobilenetv2_cifar100_m2_v2_awp0p002_cr0p50` 保存完整轻量源码。**八项全部高于历史综合最佳0624 tm010_repeat，也全部高于同期C100-M2；相对历史八项依次提高.95/.82/.39/.35/.91/.42/.56/.41个百分点，八项均值提高.60个百分点。七项攻击指标全部高于论文，但Clean仍低.72个百分点。用户选定其作为当前CIFAR-100 MobileNet最佳源码。**

对应的[活动入口](../../CIARD_Expansion_mobilenetv2_cifar100/README.md)与本目录使用同一份已评测Python源码。

## 来源与固定配置

- 实验身份：`mobilenetv2_cifar100_m2_v2_awp0p002_cr0p50`；prefix=`Cifar100_MobileNetV2_0909v1_m2_v2_awp0p002_cr0p50`，原身份及历史0909v1 prefix保持不变。
- CIFAR-100：50,000张训练 / 10,000张测试，100类，300 epochs、训练seed=0、全局batch=128，双卡满batch每卡64张。
- `training_views=2`、`awp_gamma=.002`、`consistency_weight=.50`、`method_start=120`、`method_warmup=40`、`consistency_temperature=.5`；第121轮启用双视图/新方法，第160轮达到完整强度。
- 保留M2基础配方：push=.05、push_T=5、warmup80；clean CE=.05、gate tau=2、start120/warmup80；teacher margin=.010、start140/warmup80、tau=2、cap=2；teacher-margin的clean/adv/conflict/per-sample gates均关闭，EMA decay=.999，使用普通反传。
- 鲁棒教师WRN-70-16：`models/cifar100_linf_wrn70-16_without.pt`；自然教师WRN-22-6：`models/cifar100_wrn_22_6_finetuned_best.pth`，保留CIFAR-100模型内归一化和自然教师特征接口。

KD-AWP沿用自然/对抗KD代理梯度扰动卷积/线性权重；两视图各自执行原裁剪/翻转及PGD-10（预算8/255、步长2/255），完整原损失取均值，再加入 `lambda*r*mean(JS(softmax(z_adv1/.5),softmax(z_adv2/.5)))/100`。分母来自类别数，方法实现保持来源原样。

## 与论文 CIARD baseline 对比

单位为%；括号为本结果减论文的百分点差值。Clean和白盒取[补充材料 Table 2](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Lu_CIARD_Cyclic_Iterative_ICCV_2025_supplemental.pdf)的CIFAR-100列，黑盒取[主文 Table 6](https://openaccess.thecvf.com/content/ICCV2025/papers/Lu_CIARD_Cyclic_Iterative_Adversarial_Robustness_Distillation_ICCV_2025_paper.pdf)的CIFAR-100 / Robust列；本次依据已核验的本地论文与实验结果。

| 参数设置 | Clean | 白盒 FGSM | 白盒 PGDsat | 白盒 PGDtrades | 白盒 CW | 黑盒 PGDtrades | 黑盒 Square | 黑盒 CW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 论文 CIARD baseline | 66.72 | 33.56 | 27.02 | 28.95 | 25.54 | 42.70 | 50.85 | 42.85 |
| 历史最佳 0624 tm010_repeat | 65.05 (-1.67 pp) | 33.35 (-0.21 pp) | 27.81 (+0.79 pp) | 29.54 (+0.59 pp) | 25.99 (+0.45 pp) | 43.74 (+1.04 pp) | 51.74 (+0.89 pp) | 42.72 (-0.13 pp) |
| 0911v1 C100-M1（seed 0） | 66.00 (-0.72 pp) | 34.17 (+0.61 pp) | 28.20 (+1.18 pp) | 29.89 (+0.94 pp) | 26.90 (+1.36 pp) | 44.16 (+1.46 pp) | 52.30 (+1.45 pp) | 43.13 (+0.28 pp) |

### 辅助均值比较

| 参数设置 | 白盒均值 | 黑盒均值 | 7项攻击均值 | 8项综合均值 |
| --- | ---: | ---: | ---: | ---: |
| 论文 CIARD baseline | 28.77 | 45.47 | 35.92 | 39.77 |
| 历史最佳 0624 tm010_repeat | 29.17 (+0.41 pp) | 46.07 (+0.60 pp) | 36.41 (+0.49 pp) | 39.99 (+0.22 pp) |
| 0911v1 C100-M1 | 29.79 (+1.02 pp) | 46.53 (+1.06 pp) | 36.96 (+1.04 pp) | 40.59 (+0.82 pp) |

AutoAttack单独记录：**24.59%**。补充材料的AA表不是CIFAR-100基线，不套用其他数据集阈值。均值不含AA，分别为4项白盒、3项黑盒、7项攻击与含Clean的8项等权平均；均值和差值先计算再舍入，不代表联合最坏情况准确率。

## 完成证据

- 训练作业132783、评测作业133466均已核验 `COMPLETED / 0:0`；固定EMA best epoch=240，完整10,000张测试，未改选WA或拼接不同权重成绩。
- 原始实验：`/home/lixidong25/mycode/CIARD_Expansion/run/0911v1/mobilenetv2_cifar100_m2_v2_awp0p002_cr0p50`。
- 原权重：`model/Cifar100_MobileNetV2_0909v1_m2_v2_awp0p002_cr0p50/student_best.pth`；该目录保留`training_complete.json`及`eval_best_0909v1_133466.json`。
- checkpoint SHA256：`2d9b8be81da8631115eb4ec9a96181863a40fe1d2b8a895fad0a7355878f3751`。
- evaluator SHA256：`6c030914c7315c2aa757dfa466c11ee2d1768cb636b501a2ec6b7941470161d7`。
- 原始日志：`logs/train_stdout_132783.log`与`logs/eval_best_stdout_133466.log`，以及相应Slurm out/err；本地完整报告为`结果分析/0911v1_cifar100_结果分析.md`。
- 源码与同步文件哈希见[SYNC_MANIFEST.json](../../SYNC_MANIFEST.json)，论文对照汇总见[总README](../../README.md)。
- 历史对照保留在本地`CIARD_Expansion_before0709/CIARD_Expansion0624_teacher_margin_gate_variants/cifar100_mobilenetv2_tm010_repeat`：eval121548、epoch250；历史七组完整结果中七/八项均值最高，单视图，无新增AWP/一致性。

## 使用与协议说明

本目录包含24份Python、原requirements和两份Slurm模板；Python/CFG/prefix/评测主体与来源逐字一致。脚本仅适配本目录工作/日志路径及作业名，保留rtx4090 / aias-compute-2，训练2×4090、评测1×4090、4CPU/16GB。

源码包不包含data/models资源链接、学生/教师权重、model/logs、训练完成记录或缓存，因此不是可直接提交的实验。已完成实验仍保留在原run目录，无需因本次备份重训或重评。未来复跑需复制到新的独立run目录，设置唯一prefix和匹配路径，准备公共资源及输出目录，所有GPU任务由用户手动提交。

依赖沿用已验证的ciard环境（Python3.8.20、torch1.10.0+cu113、torchvision0.11.1+cu113），CIFAR-100还使用该环境已安装的RobustBench，教师架构文件SHA256由run_checks.py锁定；原requirements保持原样，本次没有安装或升级依赖。MobileNet原有CIFAR-10辅助工具仅作为历史源码保留，本批入口为CIARD.py和attack_eval.py。

沿用test-loader的`(Clean+PGD proxy)/2`选择EMA best，存在测试集选择偏差；训练仅seed0，历史随机攻击未全部显式固定seed。PGDtrades评测20步、步长.003继承官方CIARD实现，与论文文字2/255不同；训练PGD-10仍为2/255。单次结果与论文数值的比较不等于严格同参数复现或多seed稳定性证明。

# CIFAR-100 / ResNet-18 — 0914v1 C100-R1 已评测活动入口

2026-09-17：按用户指定，从`run/0914v1/resnet18_cifar100_natorig_awp0p002_cr0p50`同步已测完整轻量源码与结果。**七项攻击全部高于论文，Clean=63.53%，仍低2.20个百分点。相对旧教师0911v1 R1，Clean提高0.53个百分点，但FGSM/白盒PGDtrades分别降低0.06/0.12个百分点；作为用户选定主线，不称为逐项最优。**

[0911v1备份](../best_backup/resnet18_cifar100_0911v1/README.md)保留旧教师版本，不是本目录新教师源码的副本。

## 来源与固定配置

- 实验身份：`resnet18_cifar100_natorig_awp0p002_cr0p50`；prefix=`Cifar100_ResNet18_0914v1_natorig_awp0p002_cr0p50`，与已测来源保持一致。
- CIFAR-100：50,000张训练/10,000张测试，100类，300 epochs、seed=0、全局batch=128，训练2×4090，满batch每卡64张。
- 双视图：training_views=2、AWP γ=.002、一致性λ=.50、method_start=120、method_warmup=40、consistency_temperature=.5；第121轮启用，第160轮达到完整强度。
- ResNet保留split KD：target=.25、non-target=0、reference=.20、start120/warmup40；push=.081740、push_T=5、warmup80；clean CE=.036067、gate tau=.682852、start158/warmup116；teacher margin=.011316、start120/warmup89、tau=1.124788、cap=1.428932；PCGrad从120轮启用，EMA=.999642。WA收尾保留，最终评测仍固定EMA student_best。
- 训练PGD-10，预算8/255、步长2/255；保留KD-AWP、两视图原损失均值及按100类归一化的一致性项。训练及评测算法原样同步。

## 自然教师更换

自然教师WRN-22-6从旧`models/cifar100_wrn_22_6_finetuned_best.pth`更换为原始包`models/nat_teacher_checkpoint/cifar100_wrn_22_6.pth`。新SHA256：`c91c5bf8b5f6c74c427d9a88815c00f98b73a4d1109fb4508b985ae86919e935`；旧SHA256：`cac8aca0c71e842e958bb49ca7b00fd729731ff465d4f50eb0ec2e53a935cecc`。配置与run_checks预检查绑定同一新权重，没有旧权重回退。

4090/3090诊断作业133714/133741均成功，17项逐样本分类结果一致；现有CIFAR-100处理下旧/新教师Clean为56.62%/76.49%。输入保持raw [0,1]与模型内CIFAR-100归一化，不增加外部Normalize。旧微调checkpoint退化原因尚未定位；本组为“新教师＋原参数”，不将换教师收益归因为新增方法创新。

鲁棒教师仍为WRN-70-16：`models/cifar100_linf_wrn70-16_without.pt`，SHA256 `3114df6b9d5adf9f275e8fea5a91b71c61df78a568ad31544b6950807d595c8c`，保留循环更新。其诊断Clean=60.86%与论文63.56%的差距尚未解释。CIFAR-10教师保持原配置。诊断原始证据在本地`结果分析/0914v1_教师诊断_4090与3090对比.md`及对应run结果中，发布摘要与hash见[SYNC_MANIFEST.json](../SYNC_MANIFEST.json)。

## 与论文 CIARD baseline 对比

单位为%，括号为本结果减论文的百分点差值。Clean/白盒取[补充材料Table 1](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Lu_CIARD_Cyclic_Iterative_ICCV_2025_supplemental.pdf)的CIFAR-100列；黑盒取[主文Table 5](https://openaccess.thecvf.com/content/ICCV2025/papers/Lu_CIARD_Cyclic_Iterative_Adversarial_Robustness_Distillation_ICCV_2025_paper.pdf)的CIFAR-100 / Robust列。每行使用同一个固定checkpoint。

| 参数设置 | Clean | 白盒 FGSM | 白盒 PGDsat | 白盒 PGDtrades | 白盒 CW | 黑盒 PGDtrades | 黑盒 Square | 黑盒 CW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 论文 CIARD baseline | 65.73 | 34.47 | 28.05 | 29.45 | 24.43 | 42.29 | 49.76 | 41.44 |
| 0914v1 C100-R1（seed 0） | 63.53 (-2.20 pp) | 35.37 (+0.90 pp) | 30.91 (+2.86 pp) | 32.02 (+2.57 pp) | 27.72 (+3.29 pp) | 43.35 (+1.06 pp) | 51.65 (+1.89 pp) | 42.27 (+0.83 pp) |

| 参数设置 | 白盒均值 | 黑盒均值 | 7项攻击均值 | 8项综合均值 |
| --- | ---: | ---: | ---: | ---: |
| 论文 CIARD baseline | 29.10 | 44.50 | 35.70 | 39.45 |
| 0914v1 C100-R1 | 31.50 (+2.40 pp) | 45.76 (+1.26 pp) | 37.61 (+1.91 pp) | 40.85 (+1.40 pp) |

AutoAttack单独记录：**25.70%**。补充材料AA表是CIFAR-10，不套用为CIFAR-100基线。均值不含AA，先计算再舍入，不代表联合最坏情况准确率。

## 完成证据

- 训练133834、评测134513均为`COMPLETED / 0:0`，EMA best epoch=200；训练实际在compute-2使用双4090，评测通过节点覆盖在compute-2使用单4090。
- 原始实验：`/home/lixidong25/mycode/CIARD_Expansion/run/0914v1/resnet18_cifar100_natorig_awp0p002_cr0p50`。
- 固定权重：`model/Cifar100_ResNet18_0914v1_natorig_awp0p002_cr0p50/student_best.pth`；同目录保留training_complete.json及`eval_best_0909v1_134513.json`。JSON文件名0909v1沿用旧命名，实验身份实际属于0914v1。
- checkpoint SHA256：`9fb11593e91c69fd82e1d19b2838bfd7f511274579f471d63e745ecfc406fd72`。
- evaluator SHA256：`f13fcffcde227994c32c920d3bd6eb325c8e134482278e8c742e2d90cd4fea9c`。
- 原始日志：`logs/train_stdout_133834.log`和`logs/eval_best_stdout_134513.log`，附Slurm out/err；本地完整报告为`结果分析/0914v1_cifar100_结果分析.md`。
- 已核验100类checkpoint CPU严格加载、训练完成标记、源码/评测器/权重hash、完整10k日志与JSON/正确数及Slurm输出一致性。来源与同步文件hash见[SYNC_MANIFEST.json](../SYNC_MANIFEST.json)，汇总见[总README](../README.md)。

## 使用与协议说明

本目录包含21份Python、原requirements及三份Slurm模板；Python、CFG、prefix和评测实现与已测run逐字一致。脚本只适配工作/日志绝对路径与job-name，默认compute-4、rtx4090、训练2卡/评测1卡、4CPU/16GB。另保留train_4090_compute2.sbatch备用训练入口，与默认入口共享prefix和输出，二选一。两组已完成评测实际使用compute-2；模板默认节点不等于历史实际执行节点。compute-2仍需注意实际7卡与GRES登记8卡的差异，保留原CUDA设备检查。

源码包不包含data/models资源链接、权重、model/logs或训练完成记录，不能直接提交为现成实验。后续复跑应复制至新的独立run目录，设置唯一prefix、匹配路径及公共资源/输出目录；全部GPU作业由用户手动提交。既有结果留在原run，无需因同步重新训练或评测。

依赖沿用已验证ciard环境（Python3.8.20、torch1.10.0+cu113、torchvision0.11.1+cu113及RobustBench），教师架构文件hash由run_checks锁定；本次未安装或升级依赖。历史辅助工具原样保留，当前入口为CIARD.py和attack_eval.py。

沿用test-loader的(Clean+PGD proxy)/2选择EMA best，存在测试集选择偏差；历史随机攻击未全部显式固定seed。评测预算8/255，PGDsat20步/2/255，PGDtrades20步/.003，CW30步/2/255，Square100queries；黑盒PGDtrades/CW由WRN-70-16生成迁移样本，Square查询学生。PGDtrades步长.003继承官方实现，与论文文字2/255存在差异。本次与论文公布值的单seed对照不等于严格同协议或统计显著性证明。

# CIFAR-10 / ResNet-18 0909v1 R5 已评测源码备份

2026-09-10：按用户指定，从 `run/0909v1/resnet18_g7_v2_awp0p002_cr0p50` 保存独立轻量备份。**R5 相对本批 G7 控制 R1 八项全部提高；相对论文七项提高，FGSM 仍低 0.12 个百分点，尚未实现论文八项全面超越。** 既有三个备份目录保持原样。

## 来源与配置

- 本批身份：R5，prefix=`Cifar10_ResNet18_0909v1_g7_v2_awp0p002_cr0p50`。
- 0906v2 G7 底座：split target=0.25、nontarget=0、reference=0.20，target mix start=120/warmup=40；nontarget=0 保留教师对抗条件分布，不关闭非目标蒸馏。
- 新方法：`training_views=2`、`awp_gamma=0.002`、`consistency_weight=0.5`、`method_start=120`、`method_warmup=40`、`consistency_temperature=0.5`。第121轮启用新分支，γ/λ第160轮达到全强度。
- 原损失：push=0.081740、push_T=5；clean CE=0.036067、gate tau=0.682852、start=158/warmup=116；teacher margin=0.011316、start=120/warmup=89、tau=1.124788、cap=1.428932；保留 teacher-margin PCGrad（start=120）和 EMA decay=0.999642。
- CIFAR-10：50,000张训练 / 10,000张测试，300 epochs、batch128、训练seed0；原双教师和学生结构不变。两视图各自执行原裁剪/翻转及PGD-10（8/255、2/255）。

KD-AWP 使用原自然KD与对抗KD目标的代理梯度，对卷积/线性权重施加 `γ*r*||w||*g/(||g||+1e-12)` 扰动；在扰动权重计算完整学生梯度，恢复权重后再SGD/EMA。两视图完整原损失取均值，并加 `λ*r*mean(JS(softmax(z_adv1/0.5),softmax(z_adv2/0.5)))/10`，两侧有梯度，不乘温度平方。方法归因和细节见[总README](../../README.md)。

## 与论文 CIARD baseline 对比

单位为百分比，括号为“R5减论文”的百分点差值；均值和差值先计算、再四舍五入。Clean和四项白盒取[补充材料 Table 1](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Lu_CIARD_Cyclic_Iterative_ICCV_2025_supplemental.pdf)，黑盒取[主文 Table 5](https://openaccess.thecvf.com/content/ICCV2025/papers/Lu_CIARD_Cyclic_Iterative_Adversarial_Robustness_Distillation_ICCV_2025_paper.pdf)的Robust列，AA取补充Table 5；本次依据项目已有本地PDF核对。

| Metric | Paper CIARD baseline | 0909v1 R5 (Δ vs baseline) |
| --- | ---: | ---: |
| Clean | 88.87 | 89.07 (+0.20) |
| White-box FGSM | 61.88 | 61.76 (-0.12) |
| White-box PGDsat | 51.70 | 52.34 (+0.64) |
| White-box PGDtrades | 54.46 | 55.01 (+0.55) |
| White-box CW | 50.61 | 51.77 (+1.16) |
| Black-box PGDtrades | 66.28 | 67.16 (+0.88) |
| Square (query-based) | 80.03 | 80.87 (+0.84) |
| Black-box CW | 64.79 | 65.71 (+0.92) |
| AutoAttack (separate) | 48.88 | 49.35 (+0.47) |
| Seven-attack mean | 61.39 | 62.09 (+0.70) |
| Eight-metric mean | 64.83 | 65.46 (+0.63) |

R5相对同期R1，按前八行顺序分别提高 +0.16 / +0.46 / +0.40 / +0.52 / +0.24 / +0.54 / +0.67 / +0.57 pp；该对照使用同批完整评测，R1的checkpoint与历史0906v2 G7相同。R5将FGSM论文差距从0.58缩至0.12pp。以上均值为描述性等权平均，不是联合最坏情况准确率，不替代逐项达标。

## 完成证据与使用

- 训练作业132648、评测作业132743：均已核验Slurm `COMPLETED / 0:0`；训练至300轮，固定EMA best为epoch230。
- 原始权重：`/home/lixidong25/mycode/CIARD_Expansion/run/0909v1/resnet18_g7_v2_awp0p002_cr0p50/model/Cifar10_ResNet18_0909v1_g7_v2_awp0p002_cr0p50/student_best.pth`。
- checkpoint SHA256：`865a125f1c82c68e7f3885674b0d72900a1ed7f0182689367f0210d1ed2e021a`。
- evaluator SHA256：`26277d6891cd0b6d76050b630a81720cfcae2269caa2d45590bc37c3b76a4164`。
- 同一model目录保留`training_complete.json`与`eval_best_0909v1_132743.json`；实验logs目录保留`train_stdout_132648.log`、`eval_best_stdout_132743.log`及Slurm日志。完整本地报告：`结果分析/0909v1_结果分析.md`。
- 20份Python和requirements与R5逐字一致；两份Slurm模板仅适配本备份目录路径及作业名称。来源与文件哈希见[同步清单](../../SYNC_MANIFEST.json)。

本备份只含源码、脚本和说明，不包含权重、日志、训练完成记录、data/models资源链接或缓存。复跑时复制到新的独立run目录，设置唯一prefix及匹配评测/脚本路径，准备公共资源和输出目录；本目录不是可直接提交的实验，所有GPU任务仍由用户手动提交。

沿用test-loader的`(Clean+PGD proxy)/2`选择EMA，存在测试集选择偏差；训练仅seed0，历史随机攻击未全部显式定seed，PGDtrades步长0.003与论文文字2/255不同。结果是历史兼容评测下对论文公布数值的对照，不宣称严格同参数复现或多seed稳定提升。

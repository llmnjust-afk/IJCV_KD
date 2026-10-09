# CIFAR-10 / ResNet-18 — 1007 C_s1 已评测论文讨论入口

**2026-10-09按用户选择更新。C_s1在完整10000张测试上八项逐项严格高于论文，AA为49.31%。** 本次报告同一个固定252轮EMA：Clean89.08%（+0.21pp）、FGSM62.10%（+0.22pp），八项均值65.41%。来源为R4配方的1007 C_s1，seed1、AWP内层自然KD系数η恒定1。

[返回1008总导航](../README.md) · [中文方法改进说明](../CIARD扩展方法改进说明.md) · [本次来源清单](../SOURCE_MANIFEST.json) · [本地更新核验](../../local_r10_cs1_20261009/verification.json)

本目录复制已测C_s1的计算源码、固定配置、依赖、脚本和成功日志，保留来源字节，仅说明文件适配阅读入口。本次更新已获用户授权同步至GitHub。**本目录用于阅读与追溯已有结果，不能直接提交作业。** 数据、教师和学生权重、资源软链接及缓存未打包，原run保持冻结。

## 来源与固定配置

方法沿0917 R4配方，经0919 E5形成1007 C_s1；原R4源码、结果及R5对比保留于[process历史入口](../process/CIARD_Expansion_resnet18_cifar10/README.md)和[原R4备份](../process/best_backup/resnet18_cifar10_0917v1_r4/README.md)。当前来源为[1007 C_s1原run（仅本地）](../../../../run/1007-cifar10-r4-awpschedule-v1/resnet18_cifar10_c_s1/README.md)。

VARIANT_NAME=`resnet18_cifar10_c_s1`，prefix=`Cifar10_ResNet18_1007_r4_awpschedule_v1_C_s1`，recipe=`C`。AWP gamma=.002、consistency_weight=.75；η初值与终值均1，调度接口不改变本组实际η。保留split KD target=.25/non-target=0/reference=.20、push=.081740、clean CE=.036067、teacher margin=.011316、margin PCGrad及学生EMA=.999642。

完整50,000张训练、batch128、seed1，Python／NumPy／Torch均采用本组seed。保留双视图、method_start120／warmup40及consistency_temperature=.5；从头训练到252轮，沿原300轮学生／教师学习率公式，不压缩日程。训练不构造test loader，不做周期测试选best、WA或alpha sweep；正式评测固定第252轮最后更新后的EMA `student_epoch252.pth`。完整CFG见[CIARD.py](CIARD.py)，文件来源与hash见[清单](../SOURCE_MANIFEST.json)。

教师保持raw输入的WRN-34-10与ResNet-56；训练为PGD-10、L∞预算8/255、步长2/255。完整评测保留历史PGDsat20步/2/255、PGDtrades20步/.003、CW30步/2/255、Square100 queries与AA；PGDtrades/CW黑盒项由鲁棒教师生成迁移样本，Square查询学生。

## 已完成测试结果

全部指标来自本组同一个预先固定252轮EMA、完整10000张测试。单位为%；差值为C_s1减原论文CIARD的百分点，均值不含AA，从整数正确数计算后舍入。论文口径见[总README](../README.md)。

| 指标 | 原论文 CIARD | C_s1 | 差值 pp |
|---|---:|---:|---:|
| Clean | 88.87 | 89.08 | +0.21 |
| 白盒 FGSM | 61.88 | 62.10 | +0.22 |
| 白盒 PGDsat | 51.70 | 52.37 | +0.67 |
| 白盒 PGDtrades | 54.46 | 54.90 | +0.44 |
| 白盒 CW | 50.61 | 51.54 | +0.93 |
| 黑盒 PGDtrades | 66.28 | 67.16 | +0.88 |
| Square 查询攻击 | 80.03 | 80.63 | +0.60 |
| 黑盒 CW | 64.79 | 65.47 | +0.68 |
| AutoAttack 单列 | 48.88 | 49.31 | +0.43 |
| 七项攻击均值 | 61.39 | 62.02 | +0.63 |
| 八项均值 | 64.83 | 65.41 | +0.58 |

C_s1是测试后选定的已测单次结果；同配方C的两个seed中1/2达到八项及AA门槛，完整记录见[1007结果报告（仅本地）](../../../../结果分析/1007-cifar10-r4-awpschedule-v1_结果分析.md)。本组η恒定1，达标不作为动态η调度的贡献。固定终点和配方参考过历史测试；历史随机攻击未全部显式定seed，不能据此宣称多seed稳定全胜或严格同协议复现。

## 完成证据

- 训练／评测为 **138301／138611**，均为`COMPLETED/0:0`；九项正确数、日志与结果JSON一致。
- 固定权重：[student_epoch252.pth（仅本地）](../../../../run/1007-cifar10-r4-awpschedule-v1/resnet18_cifar10_c_s1/model/Cifar10_ResNet18_1007_r4_awpschedule_v1_C_s1/student_epoch252.pth)，SHA256：`f57b1e8cb3cfd696ed004a2f5766527e2b6dc4734bb0065d3f46ebb41dbc0754`。
- 结果：[eval_best_0909v1_138611.json（仅本地）](../../../../run/1007-cifar10-r4-awpschedule-v1/resnet18_cifar10_c_s1/model/Cifar10_ResNet18_1007_r4_awpschedule_v1_C_s1/eval_best_0909v1_138611.json)；0909v1仅为保留的历史文件命名，本结果身份为1007 C_s1。
- evaluator SHA256：`7b47fa319fa6df83a482234f9cdfbd92728f711e7747f63b073cbdb3c84e68e9`。权重与训练完成记录、最终EMA状态指纹、源码／脚本／manifest及评测结果已核验，见[1007完成核验（仅本地）](../../../../run/1007-cifar10-r4-awpschedule-v1/preparation/results_20261009/verification.json)。
- 包内日志：[训练138301](logs/train_stdout_138301.log)、[评测138611](logs/eval_stdout_138611.log)，以及[Slurm out/err](logs/slurm/)。6份文件共1,911,824字节，stdout与对应out逐字节一致，两份err为空；未复制运行锁，四入口日志总量见[总README](../README.md)。

## 源码与运行绑定

本目录共31份文件：21份Python、2份sbatch、requirements.txt、本文和6份成功日志。计算源码、依赖与脚本均来自已测run且逐字一致；[training_state.py](training_state.py)保留训练状态指纹实现。训练与完整评测入口为[CIARD.py](CIARD.py)、[attack_eval.py](attack_eval.py)，配套[train_4090.sbatch](train_4090.sbatch)和[eval_4090_fixed.sbatch](eval_4090_fixed.sbatch)。

源码与脚本保留原 `run/1007-cifar10-r4-awpschedule-v1/resnet18_cifar10_c_s1` 路径、prefix、父manifest及固定权重绑定；资源仍为rtx4090、aias-compute-4、单4090、4CPU/16GB。复制到1008不改变这些运行身份，不能在本目录直接执行，也不能向已完成原run重提。

未来复跑须另建独立run，重新准备唯一身份、公共资源链接、输出目录和配套校验，所有GPU作业由用户手动提交。本次仅整理已有已测源码与日志，无需重训或重评。

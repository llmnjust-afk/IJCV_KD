# CIFAR-10 / ResNet18 — 0909v1 R5 候选入口

2026-09-09：按用户确认，从 `run/0909v1/resnet18_g7_v2_awp0p002_cr0p50` 同步代码。**这是待完整评测的候选源码，不是已验证的性能最佳版本。** 原实验已有用户训练日志，尚无完整测试结果；本源码目录没有权重或运行产物。

## 来源、配置与改进

- 原始配方：0906v2 G7：split target=.25 / nontarget=0 / reference=.20，push=.081740，clean CE=.036067，teacher-margin PCGrad，EMA=.999642。
- 冻结参考：`origin_code/0909v1/IJCV_KD/best_backup/resnet18_cifar10_0906v2`。包内 `best_backup` 所有文件保持原样。
- 实验身份：`resnet18_g7_v2_awp0p002_cr0p50`；prefix：`Cifar10_ResNet18_0909v1_g7_v2_awp0p002_cr0p50`。
- 新机制固定为 `training_views=2`、`awp_gamma=.002`、`consistency_weight=.5`、`method_start=120`、`method_warmup=40`、`consistency_temperature=.5`。
- 从头300轮、batch128、seed0、50k CIFAR-10训练；前120轮保留原分支，121轮渐进启用，160轮完整启用。

KD-AWP 使用原自然KD与对抗KD目标的代理梯度扰动卷积/线性权重，BN和bias不扰动；在扰动权重计算完整训练目标并反向传播，恢复正常权重后SGD及EMA。双视图各自进行原裁剪/翻转与PGD-10（8/255、2/255）；完整原损失取均值，再加类别尺度归一化的对抗预测JS一致性。两个机制的详细公式、已有文献归因及实验对照见[包README](../README.md)。

历史 G7 七项超过论文，FGSM61.30低于61.88，AA49.25；这些是 G7 的结果，不是本 R5 候选的成绩。

## 固定评测与证据边界

固定目标为 `model/Cifar10_ResNet18_0909v1_g7_v2_awp0p002_cr0p50/student_best.pth`，沿用历史 `(Clean + PGD proxy)/2` 选择EMA，训练成功后完整评测10k测试集。同一个 EMA checkpoint 八项严格超过88.87/61.88/51.70/54.46/50.61/66.28/80.03/64.79，且AA≥48.88。

`attack_eval.py`、训练及完成检查Python源码与对应run候选逐字一致，清单见[SYNC_MANIFEST.json](../SYNC_MANIFEST.json)。保留原八项与AA攻击参数、顺序和输出；评测前要求对应训练Slurm成功、TRAIN_COMPLETE及来源/权重hash一致。没有复制训练完成记录，不能用此目录直接读取run中的完成标记。

test-loader选模存在测试选择偏差，随机攻击未全部显式定seed；历史PGDtrades步长为.003，与论文文字2/255不同。CPU/静态检查不能证明性能提升。

## 源码包与运行入口

本目录只保存轻量代码、脚本和说明，不含 data/models 资源链接、学生/教师权重、model、logs 或缓存。40份同步Python文件分布于两个模型目录；各组源码、CFG和prefix与原实验一致。两份Slurm脚本仅适配此目录绝对路径和独立作业名称，资源为 rtx4090 / aias-compute-4 / gpu:4090:1 / 4CPU / 16GB。

这些脚本是源码模板，本目录当前不是可直接提交的实验。未来复跑应复制到新的独立run目录，设置唯一prefix、评测路径、脚本工作目录和日志目录，并准备公共资源链接与空输出目录。依赖清单采用R5已核验的ciard环境版本，不需要为本次同步安装或升级依赖。

现有批次继续使用下列独立实验；训练已由用户提交，不需要因本次同步重复训练：

```text
/home/lixidong25/mycode/CIARD_Expansion/run/0909v1/resnet18_g7_v2_awp0p002_cr0p50
```

仅在该组训练成功后，用户手动提交完整评测：

```bash
sbatch /home/lixidong25/mycode/CIARD_Expansion/run/0909v1/resnet18_g7_v2_awp0p002_cr0p50/eval_4090_best.sbatch
```

训练日志位于该run目录的 `logs/train_stdout_<job>.log`，评测日志为 `logs/eval_best_stdout_<job>.log`；汇总通过本地 `run/0909v1/summarize_results.py` 在测试完成后生成。本次没有提交、取消、重启训练或推送GitHub。

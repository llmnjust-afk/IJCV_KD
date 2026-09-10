# CIFAR-10 / MobileNetV2 — 0909v1 M2 已评测活动入口

2026-09-11：活动源码与 `run/0909v1/mobilenetv2_best_v2_awp0p002_cr0p50` 的20份Python逐字一致。**M2已完成10k测试，相对同期M1七项提高、黑盒CW回落；对论文也仅七项提高，尚非全面改进。** 本目录不含权重或运行产物，完整对照见[总README](../README.md)。

## 来源、配置与改进

- 原始配方：已验证 tm010-repeat：push=.05，clean CE=.05，margin=.010 / start140 / warmup80，EMA=.999；保留 MobileNet 普通反传，不启用 ResNet split KD 或 PCGrad。
- 冻结参考：`origin_code/0909v1/IJCV_KD/best_backup/mobilenetv2_cifar10`。该MobileNet冻结参考保持原样。
- 实验身份：`mobilenetv2_best_v2_awp0p002_cr0p50`；prefix：`Cifar10_MobileNetV2_0909v1_best_v2_awp0p002_cr0p50`。
- 新机制固定为 `training_views=2`、`awp_gamma=.002`、`consistency_weight=.5`、`method_start=120`、`method_warmup=40`、`consistency_temperature=.5`。
- 从头300轮、batch128、seed0、50k CIFAR-10训练；前120轮保留原分支，121轮渐进启用，160轮完整启用。

KD-AWP 使用原自然KD与对抗KD目标的代理梯度扰动卷积/线性权重，BN和bias不扰动；在扰动权重计算完整训练目标并反向传播，恢复正常权重后SGD及EMA。双视图各自进行原裁剪/翻转与PGD-10（8/255、2/255）；完整原损失取均值，再加类别尺度归一化的对抗预测JS一致性。两个机制的详细公式、已有文献归因及实验对照见[包README](../README.md)。

已验证 tm010-repeat 的八项优势属于原配方。同期M1是单视图/AWP0/一致性0的复跑控制，八项均值64.27%，checkpoint SHA256与归档best一致；它没有启用本次新增机制。**M2与ResNet R5仅共用新增方法及其参数，各自基础损失和优化配方保留。** M2八项均值64.95%，比M1高0.68pp，AA47.15%比M1高0.91pp；但黑盒CW65.95%比M1低0.23pp、比论文低0.17pp，不能据此替换旧best的八项全胜结论。原顶层push=.075的0903候选已由本M2源码替换。

## 固定评测与证据边界

固定目标为 `model/Cifar10_MobileNetV2_0909v1_best_v2_awp0p002_cr0p50/student_best.pth`，沿用历史 `(Clean + PGD proxy)/2` 选择EMA，训练成功后完整评测10k测试集。八项参照89.51/59.10/47.67/50.71/46.88/66.66/80.01/66.12，AA参照46.31；同时报告M2−M1，不能用仍超过论文代替优于当前最佳。

`attack_eval.py`、训练及完成检查Python源码与对应run候选逐字一致，清单见[SYNC_MANIFEST.json](../SYNC_MANIFEST.json)。保留原八项与AA攻击参数、顺序和输出；评测前要求对应训练Slurm成功、TRAIN_COMPLETE及来源/权重hash一致。没有复制训练完成记录，不能用此目录直接读取run中的完成标记。

test-loader选模存在测试选择偏差，随机攻击未全部显式定seed；历史PGDtrades步长为.003，与论文文字2/255不同。CPU/静态检查不能证明性能提升。

## 源码包与运行入口

本目录只保存轻量代码、脚本和说明，不含 data/models 资源链接、学生/教师权重、model、logs 或缓存。40份同步Python文件分布于两个模型目录；各组源码、CFG和prefix与原实验一致。两份Slurm脚本仅适配此目录绝对路径和独立作业名称，资源为 rtx4090 / aias-compute-2 / gpu:4090:1 / 4CPU / 16GB。

这些脚本是源码模板，本目录当前不是可直接提交的实验。未来复跑应复制到新的独立run目录，设置唯一prefix、评测路径、脚本工作目录和日志目录，并准备公共资源链接与空输出目录。依赖清单采用R5已核验的ciard环境版本，不需要为本次同步安装或升级依赖。

原实验已完成训练和评测，无需因本次发布重复运行：

```text
/home/lixidong25/mycode/CIARD_Expansion/run/0909v1/mobilenetv2_best_v2_awp0p002_cr0p50
```

训练132654与评测132769均已核验Slurm COMPLETED/0:0，固定EMA best epoch=250；日志为`logs/train_stdout_132654.log`、`logs/eval_best_stdout_132769.log`。原model目录保留训练完成记录及`eval_best_0909v1_132769.json`。权重/evaluator哈希见总README和同步清单；本地完整报告为`结果分析/0909v1_结果分析.md`。本次仅更新结果文档，不触发GPU任务。

本目录保留的 convert_rb_teacher.py、fast_eval.py、setup_models.sh 和 train_teacher.py 是历史辅助工具，不属于本次固定教师训练和完整评测入口；不需要执行它们。旧3090提交脚本已由4090脚本替代。

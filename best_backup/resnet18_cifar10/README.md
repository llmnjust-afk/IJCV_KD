# CIFAR-10 / ResNet-18 当前综合主线

## 身份与来源

- 配置：0703 `r18_pcgrad_optuna_transfer`
- prefix：`Cifar10_ResNet18_0703_pcgrad_optuna_transfer`
- 整理前冻结副本：`my_temp_best/CIARD_Expansion_resnet18_cifar10_v1`
- 归档后来源：`CIARD_Expansion_before0824/my_temp_best/CIARD_Expansion_resnet18_cifar10_v1`
- 证据：`结果分析/CIARD_Expansion0703_pcgrad_resnet18_variants_结果分析.md`

该配置是当前 ResNet-18 综合主线。0712 的 `v1.2 previous_best_control` 基本复现了它；Label Smoothing/Adaptive Temperature 新组合没有形成可替代的全面收益。

| Clean | FGSM | PGDsat | PGDtrades | CW | 黑盒 PGDtrades | Square | 黑盒 CW |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 88.66 | 61.33 | 52.21 | 54.76 | 51.27 | 66.60 | 80.09 | 65.02 |

白盒均值为 54.89，7 项鲁棒均值为 61.61，8 项综合均值为 64.99，AutoAttack final robust 为 49.08。结果来自 seed 0 的 `student_best.pth` 单次完整评测。

## 使用说明

本目录只保存代码，不包含历史 checkpoint 或日志。`data`、`models` 指向项目根目录公共资源，输出目录已初始化。`train_4090.sbatch` 和 `eval_4090_best.sbatch` 的路径已经改到本目录；评测前必须先生成 `model/Cifar10_ResNet18_0703_pcgrad_optuna_transfer/student_best.pth`。

所有训练和评测作业只能由用户手动提交，Codex 不得执行 `sbatch`。

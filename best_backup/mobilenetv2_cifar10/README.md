# CIFAR-10 / MobileNet-V2 当前综合主线

## 身份与来源

- 配置：0624 `cifar10_mobilenetv2_tm010_repeat`
- prefix：`Cifar10_MobileNetV2_tm010_repeat0620`
- 整理前冻结副本：`my_temp_best/CIARD_Expansion_mobilenetv2_cifar10_v1`
- 归档后来源：`CIARD_Expansion_before0824/my_temp_best/CIARD_Expansion_mobilenetv2_cifar10_v1`
- 证据：`结果分析/CIARD_Expansion0624_teacher_margin_gate_variants_结果分析.md`

它是当前“8 项全部高于论文 CIARD baseline”的稳妥主线。`v1.4 repaired_gentle` 的等权综合均值更高，但 Clean、Square 和黑盒 CW 低于 baseline，因此没有替换本目录。

| Clean | FGSM | PGDsat | PGDtrades | CW | 黑盒 PGDtrades | Square | 黑盒 CW |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 89.58 | 60.12 | 49.56 | 52.28 | 48.50 | 67.32 | 80.78 | 66.18 |

7 项鲁棒均值为 60.68，8 项综合均值为 64.29。结果来自 seed 0 的 `student_best.pth` 单次完整评测。

## 使用说明

本目录只保存代码，不包含历史 checkpoint 或日志。`data`、`models` 指向项目根目录公共资源，输出目录已初始化。`train_4090.sbatch` 和 `eval_4090_best.sbatch` 的路径已经改到本目录；评测前必须先生成 `model/Cifar10_MobileNetV2_tm010_repeat0620/student_best.pth`。

所有训练和评测作业只能由用户手动提交，Codex 不得执行 `sbatch`。

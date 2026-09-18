# Source backup index

Current user-selected snapshots, copied locally on2026-09-19. Directory dates identify the evaluated experiment batch, not a new training run.

| Current snapshot | Configuration / result |
| --- | --- |
| [mobilenetv2_cifar10_0917v1_m1](mobilenetv2_cifar10_0917v1_m1/README.md) | CIFAR-10 M1; AWP.001 / consistency.50;8/8 above paper, AA47.01%. Includes [M4 reproduction guide](mobilenetv2_cifar10_0917v1_m1/M4_REPRODUCTION.md) and both results. |
| [resnet18_cifar10_0917v1_r4](resnet18_cifar10_0917v1_r4/README.md) | CIFAR-10 R4; AWP.002 / consistency.75;7/8, FGSM−.08pp, AA49.10%. User-selected source, not all-metric winner. |
| [mobilenetv2_cifar100_0914v1_m1](mobilenetv2_cifar100_0914v1_m1/README.md) | CIFAR-100 M1; original-package natural teacher;8/8 above paper, AA24.95%. |

The current CIFAR-100 ResNet source is [0914v1 R2](../CIARD_Expansion_resnet18_cifar100/README.md), using the canonical directory without an _r2 suffix. R1 survives in the earlier origin_code/0914v1 source package and original run; its historical result is still documented. No new CIFAR-100 ResNet backup was requested in this synchronization.

## Frozen historical references

| Historical backup | Description |
| --- | --- |
| [mobilenetv2_cifar10](mobilenetv2_cifar10/README.md) | Preserved MobileNet-V2 reference. |
| [resnet18_cifar10](resnet18_cifar10/README.md) | Preserved 0703 ResNet-18 reference. |
| [resnet18_cifar10_0906v2](resnet18_cifar10_0906v2/README.md) | Completed 0906v2 G7: seven primary metrics exceed baseline; FGSM remains 0.58 points below. |
| [resnet18_cifar10_0909v1](resnet18_cifar10_0909v1/README.md) | Completed 0909v1 R5: all eight primary metrics improve over same-batch R1; seven exceed the paper baseline, with FGSM 0.12 pp below. |
| [resnet18_cifar100_0911v1](resnet18_cifar100_0911v1/README.md) | Completed CIFAR-100 C100-R1: seven attacks exceed the paper; Clean remains 2.73 pp below. Previous user-selected ResNet source (old natural teacher). |
| [mobilenetv2_cifar100_0911v1](mobilenetv2_cifar100_0911v1/README.md) | Completed CIFAR-100 C100-M1: all eight metrics improve over the historical best; seven attacks exceed the paper, with Clean 0.72 pp below. |


All six historical directories above remain byte-for-byte unchanged. Their older internal “current” labels describe their original snapshot dates; the current selection is the table at the top of this index and the repository README.

New backups include lightweight source, scripts and documentation only. Resource links, datasets, checkpoints, logs and caches are excluded. Prepare a new independent run before reproduction; only the user submits jobs. Python matches the evaluated run; wrapper paths are adapted to each package directory. Source synchronization and verification are complete; publication was authorized by the user on 2026-09-19.

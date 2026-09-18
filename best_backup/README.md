# Source backup index

| Backup | Description |
| --- | --- |
| [mobilenetv2_cifar10](mobilenetv2_cifar10/README.md) | Preserved MobileNet-V2 reference. |
| [resnet18_cifar10](resnet18_cifar10/README.md) | Preserved 0703 ResNet-18 reference. |
| [resnet18_cifar10_0906v2](resnet18_cifar10_0906v2/README.md) | Completed 0906v2 G7: seven primary metrics exceed baseline; FGSM remains 0.58 points below. |
| [resnet18_cifar10_0909v1](resnet18_cifar10_0909v1/README.md) | Completed 0909v1 R5: all eight primary metrics improve over same-batch R1; seven exceed the paper baseline, with FGSM 0.12 pp below. |
| [resnet18_cifar100_0911v1](resnet18_cifar100_0911v1/README.md) | Completed CIFAR-100 C100-R1: seven attacks exceed the paper; Clean remains 2.73 pp below. Previous user-selected ResNet source (old natural teacher). |
| [mobilenetv2_cifar100_0911v1](mobilenetv2_cifar100_0911v1/README.md) | Completed CIFAR-100 C100-M1: all eight metrics improve over the historical best; seven attacks exceed the paper, with Clean 0.72 pp below. |

The CIFAR-10 active entries retain R5 and M2; their four existing backups remain unchanged.
The CIFAR-100 source entries retain completed 0914v1 C100-R1 and C100-R2 in parallel, plus MobileNet C100-M1, all with the original-package natural teacher. The independent 0911v1 backups retain the previous sources, old natural teacher and historical results; they are not copies of the current active entries.
New backups contain source, scripts and documentation only. Resources, checkpoints and logs
are not bundled. See each README for provenance and manual preparation in a new independent run.

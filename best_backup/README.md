# Source backup index

| Backup | Description |
| --- | --- |
| [mobilenetv2_cifar10](mobilenetv2_cifar10/README.md) | Preserved MobileNet-V2 reference. |
| [resnet18_cifar10](resnet18_cifar10/README.md) | Preserved 0703 ResNet-18 reference. |
| [resnet18_cifar10_0906v2](resnet18_cifar10_0906v2/README.md) | Completed 0906v2 G7: seven primary metrics exceed baseline; FGSM remains 0.58 points below. |
| [resnet18_cifar10_0909v1](resnet18_cifar10_0909v1/README.md) | Completed 0909v1 R5: all eight primary metrics improve over same-batch R1; seven exceed the paper baseline, with FGSM 0.12 pp below. |

The active ResNet entry now uses evaluated R5; this independent backup and the earlier
backups remain available unchanged. G7 and R5 contain source, scripts and documentation only. Resources,
checkpoints and logs are not bundled. See each backup README for provenance and
manual preparation in a new independent run directory.

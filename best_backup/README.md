# Source backup index

| Backup | Description |
| --- | --- |
| [mobilenetv2_cifar10](mobilenetv2_cifar10/README.md) | Preserved MobileNet-V2 reference. |
| [resnet18_cifar10](resnet18_cifar10/README.md) | Preserved 0703 ResNet-18 reference. |
| [resnet18_cifar10_0906v2](resnet18_cifar10_0906v2/README.md) | Completed 0906v2 G7: seven primary metrics exceed baseline; FGSM remains 0.58 points below. |

G7 is added separately and does not replace the active ResNet entry or earlier
backups. It contains source, scripts and documentation only. Resources,
checkpoints and logs are not bundled. See its README for provenance and
manual preparation in a new independent run directory.

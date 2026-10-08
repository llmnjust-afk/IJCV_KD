# Source backup index

Current recommendations as of 2026-10-05. The user selected 1004 H2 for the CIFAR-100 ResNet entry; the other three selections remain unchanged. Directory dates identify the evaluated experiment batch, not the copy date. H2 is a new local snapshot dated 2026-10-05; the three earlier recommended snapshots retain their 2026-09-19 identities.

| Model / dataset | Current recommended snapshot | Result and selection boundary |
| --- | --- | --- |
| ResNet-18 / CIFAR-100 | [resnet18_cifar100_1004_h2](resnet18_cifar100_1004_h2/README.md) | 1004 H2; outer natural-binary weight=2, start=120, warmup=40. This seed-0 run passes 8/8 paper thresholds at fixed epoch-190 EMA: Clean 65.96%, FGSM 34.52%, AA 24.22%, eight-metric mean 40.67%. Minimum margin is FGSM +.05pp; this is a verified passing result, not evidence of stable or universally best performance. |
| MobileNet-V2 / CIFAR-10 | [mobilenetv2_cifar10_0917v1_m1](mobilenetv2_cifar10_0917v1_m1/README.md) | 0917v1 M1; AWP .001 / consistency .50; 8/8 above paper, AA 47.01%, eight-metric mean 64.83%. Keep the included [M4 reproduction guide](mobilenetv2_cifar10_0917v1_m1/M4_REPRODUCTION.md) and both results. |
| ResNet-18 / CIFAR-10 | [resnet18_cifar10_0917v1_r4](resnet18_cifar10_0917v1_r4/README.md) | 0917v1 R4; AWP .002 / consistency .75; 7/8, FGSM −.08pp, AA 49.10%, eight-metric mean 65.34%. User-selected source; it does not pass all eight thresholds or dominate historical R5. |
| MobileNet-V2 / CIFAR-100 | [mobilenetv2_cifar100_0914v1_m1](mobilenetv2_cifar100_0914v1_m1/README.md) | 0914v1 M1; original-package natural teacher; 8/8 above paper, Clean 66.91%, AA 24.95%, eight-metric mean 41.15%. Keep this selection; the 0926 A800 result does not replace it. |

The active CIFAR-100 ResNet directory is [CIARD_Expansion_resnet18_cifar100](../CIARD_Expansion_resnet18_cifar100/README.md), now containing H2. H2 training/evaluation jobs are 137534/137729. Full metrics and historical columns remain in the [repository README](../README.md); CIFAR-100 AA is reported separately, without an invented paper threshold.

## References to retain when considering future cleanup

- **CIFAR-10 ResNet R4 and R5:** retain both recommended R4 and [historical R5](resnet18_cifar10_0909v1/README.md). R4 has higher Clean, FGSM and black-box PGDtrades; R5 has higher other five primary metrics and AA. R5 AA 49.35% and eight-metric mean 65.46% exceed R4's 49.10% and 65.34%. Both pass 7/8, so neither is an all-metric winner.
- **CIFAR-10 MobileNet M1 and M4:** retain M1 together with its M4 guide. Both pass 8/8; M1 has higher FGSM, white-box PGDtrades, white-box CW and AA, while M4 has higher Clean, white-box PGDsat, black-box PGDtrades and Square. Black-box CW ties at 66.14%, only +.02pp above paper. M4's eight-metric mean 64.89% is higher, but its AA 46.70% is lower. A separate duplicate M4 source directory is not needed for this synchronization.
- **CIFAR-100 MobileNet M1 and 0926:** retain the selected 0914v1 M1. In the A800 comparison, the same old M1 checkpoint has AA 24.96% and eight-metric mean 41.15%; the new 0926 model has 24.84% and 41.05%. The new model improves Clean and three white-box metrics but loses white-box CW, all three black-box metrics and AA. Both pass 8/8; the newer run is a historical migration reference, not a replacement selection.
- **CIFAR-100 ResNet R2 provenance:** the 26 lightweight files, including 21 Python files, in the former 1004 active R2 entry were verified byte-for-byte identical to [the frozen 0917v1-cifar10 R2 entry](../../../0917v1-cifar10/IJCV_KD/CIARD_Expansion_resnet18_cifar100/README.md). That old package supplies the migration source mapping. No existing backup directory below is an equivalent R2 snapshot; `resnet18_cifar100_0911v1` is the old-teacher R1. Keep the mapped R2 source available, or preserve a lightweight copy before any future cleanup of that old package. R1 also remains in the earlier 0914v1 package and original run.

This index records retention priorities; no backups are deleted in this synchronization.

## Frozen historical references

| Historical backup | Description |
| --- | --- |
| [mobilenetv2_cifar10](mobilenetv2_cifar10/README.md) | Preserved MobileNet-V2 reference. |
| [resnet18_cifar10](resnet18_cifar10/README.md) | Preserved 0703 ResNet-18 reference. |
| [resnet18_cifar10_0906v2](resnet18_cifar10_0906v2/README.md) | Completed 0906v2 G7: seven primary metrics exceed baseline; FGSM remains 0.58 points below. |
| [resnet18_cifar10_0909v1](resnet18_cifar10_0909v1/README.md) | Completed 0909v1 R5: all eight primary metrics improve over same-batch R1; seven exceed the paper baseline, with FGSM 0.12 pp below. |
| [resnet18_cifar100_0911v1](resnet18_cifar100_0911v1/README.md) | Completed CIFAR-100 C100-R1: seven attacks exceed the paper; Clean remains 2.73 pp below. Previous user-selected ResNet source (old natural teacher). |
| [mobilenetv2_cifar100_0911v1](mobilenetv2_cifar100_0911v1/README.md) | Completed CIFAR-100 C100-M1: all eight metrics improve over the historical best; seven attacks exceed the paper, with Clean 0.72 pp below. |


All nine pre-existing backup directories, including the six historical directories above, remain byte-for-byte unchanged. Their older internal “current” labels describe their original snapshot dates; the current selection is the table at the top of this index and the repository README.

New backups include lightweight source, scripts and documentation only. Resource links, datasets, checkpoints, logs and caches are excluded. Prepare a new independent run before reproduction; only the user submits jobs. H2 includes 50 Python files (25 active and 25 frozen reference files), dependency information and two A800 script templates. Its run/manifest identity checks remain bound to the evaluated original run, so this snapshot cannot be submitted directly. Reproduction requires a new independent run with explicitly prepared identities, manifest, resource links and output paths; the completed original run remains frozen. Earlier snapshots retain their own documented wrapper conventions.

The 2026-09-19 publication belongs to the earlier synchronization. The user authorized GitHub publication of the verified 2026-10-05 H2 source update to `llmnjust-afk/IJCV_KD` on `main`. The earlier local-only synchronization is retained as historical evidence; this publication submits no cluster jobs.

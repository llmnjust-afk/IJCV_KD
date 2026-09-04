# IJCV_KD 0903 candidate source

This snapshot presents one selected CIFAR-10 candidate for each student model:

- `CIARD_Expansion_mobilenetv2_cifar10`: the 0624 MobileNet-V2 mainline
  with only `push_lambda` changed from `0.05` to `0.075`.
- `CIARD_Expansion_resnet18_cifar10`: the 0703 ResNet-18 PCGrad mainline
  with conservative projected clean/FGSM auxiliary gradients.

Both candidates use the existing raw WRN-34-10 robust teacher and raw ResNet-56
natural teacher. They are code-review candidates only: the 0903 jobs were
cancelled and cleaned, so no completed training, evaluation result, or new-best
claim exists.

The prior 0830 SARD/SAA/RCD implementation is no longer the active source. Its
eight-run evaluation showed no replacement for the 0624 MobileNet-V2 or 0703
ResNet-18 mainline, so the SARD-oriented guides, configs, scripts, and example
results are retained only as historical material. They must not be treated as
the current execution interface.

Each active model directory contains a fixed training candidate, its frozen
historical evaluator, and matching 3090 Slurm wrappers. The repository excludes
datasets, teacher checkpoints, logs, model outputs, and machine-local symlinks.
All Slurm jobs must be submitted manually by the user.

The `best_backup/` directory remains a separate frozen record of the previously
verified best code and is not modified by this 0903 synchronization.

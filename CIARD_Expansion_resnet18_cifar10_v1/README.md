# r18_pcgrad_optuna_transfer

Independent CIFAR-10 / ResNet-18 experiment copied from `CIARD_Expansion0703`.

## Fixed Identity

- variant: `r18_pcgrad_optuna_transfer`
- prefix: `Cifar10_ResNet18_0703_pcgrad_optuna_transfer`
- node: `aias-compute-4`
- student: `resnet18`
- seed: `0`
- epochs: `300`
- batch size: `128`
- training PGD: `10` steps, epsilon `8/255`, step size `2/255`
- weight averaging: evaluate both `student_best.pth` and `student_weight_averaged.pth`

## Purpose

0629 Optuna-like clean/black-box region transferred to PCGrad; start rounded to 120 for WA checkpoint cadence.

## Key Settings

- `teacher_margin_weight=0.011316`
- `teacher_margin_start=120`
- `teacher_margin_warmup=89`
- `teacher_margin_tau=1.124788`
- `teacher_margin_cap=1.428932`
- `teacher_margin_clean_gate=False`
- `teacher_margin_adv_gate=False`
- `teacher_margin_conflict_gate=True`
- `teacher_margin_conflict_floor=0.211797`
- `teacher_margin_per_sample_conflict=False`
- `teacher_margin_relative=False`
- `clean_ce_weight=0.036067`
- `adv_ce_weight=0.0`
- `ce_start=158`
- `ce_warmup=116`
- `clean_ce_gate_tau=0.682852`
- `push_lambda=0.08174`
- `push_T=5.0`
- `pcgrad_teacher_margin=True`
- `pcgrad_start=120`
- `weight_averaging=True`
- `wa_alpha=0.5`
- `student_ema=True`
- `student_ema_decay=0.999642`

`CIARD.py` prints the resolved training configuration as a multi-line block.
`attack_eval.py`, `attack_eval_wa.py`, `attack_eval_latest.py`, and `attack_eval_300.py` print the fixed checkpoint and attack settings.

## Commands

Run from the project root:

```bash
sbatch CIARD_Expansion_resnet18_cifar10_v1/train_4090.sbatch
```

After training creates `student_best.pth`, run:

```bash
sbatch CIARD_Expansion_resnet18_cifar10_v1/eval_4090_best.sbatch
```

After training creates `student_weight_averaged.pth`, run:

```bash
sbatch CIARD_Expansion_resnet18_cifar10_v1/eval_4090_wa.sbatch
```

To compare checkpoint selection effects after training has completed, run:

```bash
sbatch CIARD_Expansion_resnet18_cifar10_v1/eval_4090_latest.sbatch
sbatch CIARD_Expansion_resnet18_cifar10_v1/eval_4090_300.sbatch
```

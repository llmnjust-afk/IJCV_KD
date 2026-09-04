# CIFAR-10 / ResNet-18 0903 projected-auxiliary candidate

## Candidate identity

- source baseline: 0703 `r18_pcgrad_optuna_transfer`
- variant: `resnet18_paux_c015_f025_cap05`
- prefix: `Cifar10_ResNet18_0903v1_paux_c015_f025_cap05`
- robust teacher: raw WRN-34-10
- natural teacher: raw ResNet-56

The candidate retains the complete 0703 update as the reference gradient. It
adds robust-gated clean CE with weight `0.015` and an FGSM auxiliary objective
with weight `0.025`. Each new auxiliary gradient has its conflicting component
projected away, and their combined norm is capped at `0.05 × ||g_ref||` before
being added to the reference update. The FGSM branch starts at epoch 170,
warms up for 60 epochs, uses `epsilon=8/255`, and mixes CE with robust-teacher
KL at `0.5/0.5`. Its auxiliary forwards temporarily use evaluation mode so
they do not alter student or teacher BatchNorm running statistics.

This conservative configuration was selected because the 0703 baseline's
remaining deficits are Clean and FGSM, while stronger historical repair terms
often reduced PGD/CW/AutoAttack. It is an engineering hypothesis, not a
performance conclusion. All matching 0903 training attempts were cancelled
and cleaned; no completed checkpoint or evaluation exists.

## Protocol and limitations

The source is the exact training snapshot prepared in
`run/0903v1/resnet18_paux_c015_f025_cap05`. It intentionally does not retain
the 0830 SAA/RCD, label-smoothing, adaptive-temperature, or old unprotected
FGSM-repair path because the completed 0830 matrix did not improve the
mainline.

The inherited 0703 code trains on all 50,000 CIFAR-10 training samples and uses
the test loader for checkpoint selection. That historical selection bias must
be retained in any result description. `attack_eval.py` is the frozen
historical evaluator; its stochastic attacks do not all receive an explicit
seed.

## Local preparation and manual execution

The Git repository does not contain datasets, teacher checkpoints, logs, model
outputs, or machine-local symlinks. Before running, create `data` and `models`
links to the project-wide resources and create empty `logs/slurm` and `model`
directories.

The bundled wrappers target `aias-compute-01` and request one typed 3090 GPU.
They perform CUDA/data/teacher preflight and emit hash-bearing completion
markers. From the CIARD project root, the user may submit them manually:

```bash
sbatch origin_code/0903v1/IJCV_KD/CIARD_Expansion_resnet18_cifar10/train_3090.sbatch
```

Only after training is successful, the checkpoint is non-empty, and the log
contains `TRAIN_COMPLETE`:

```bash
sbatch origin_code/0903v1/IJCV_KD/CIARD_Expansion_resnet18_cifar10/eval_3090_best.sbatch
```

Codex must not submit either job.

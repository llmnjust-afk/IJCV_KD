# CIFAR-10 / MobileNet-V2 0903 candidate

## Candidate identity

- source baseline: 0624 `cifar10_mobilenetv2_tm010_repeat`
- variant: `mobilenetv2_push0075`
- prefix: `Cifar10_MobileNetV2_0903v1_push0075`
- robust teacher: raw WRN-34-10
- natural teacher: raw ResNet-56

This candidate changes only `push_lambda` from `0.05` to `0.075`. The midpoint
was selected to target the remaining black-box CW margin without making the
weak reliable-push term as aggressive as `0.10`.

This is an unvalidated candidate, not a new best result. The matching 0903
training attempts were cancelled and cleaned, so there is no completed
checkpoint or evaluation result. The known best evidence still belongs to the
0624 source configuration.

## Protocol and limitations

The source is the exact training snapshot prepared in
`run/0903v1/mobilenetv2_push0075`. It intentionally does not retain the 0830
SAA/RCD, label-smoothing, adaptive-temperature, late-clean-CE, or FGSM-anchor
training path because the completed 0830 matrix did not improve the mainline.

The inherited 0624 code trains on all 50,000 CIFAR-10 training samples and uses
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
sbatch origin_code/0903v1/IJCV_KD/CIARD_Expansion_mobilenetv2_cifar10_v1/train_3090.sbatch
```

Only after training is successful, the checkpoint is non-empty, and the log
contains `TRAIN_COMPLETE`:

```bash
sbatch origin_code/0903v1/IJCV_KD/CIARD_Expansion_mobilenetv2_cifar10_v1/eval_3090_best.sbatch
```

Codex must not submit either job.

# CIFAR-10 / MobileNet-V2 source

Fixed identity:

- student: MobileNet-V2
- robust teacher: raw WRN-34-10
- natural teacher: raw ResNet-56
- shared checkpoints: `models/model_cifar_wrn.pt` and
  `models/nat_teacher_checkpoint/cifar10_resnnet56.pth`

`CIARD.py` exposes `--sard_saa`, `--sard_rcd`, `--epochs`, `--prefix`, and
`--original_ciard`. It strict-loads both teachers, uses a fixed 45k/5k
train/validation split for checkpoint selection, and refuses to reuse a
non-empty output prefix. The resolved configuration and checkpoint identities
are printed before training. The bundled training template verifies one visible
CUDA GPU, CIFAR-10 integrity, and the exact hashes of both teachers before
starting Python; it prints `TRAIN_COMPLETE` with the best-checkpoint hash only
after a successful zero-exit training process and a non-empty checkpoint.

`attack_eval.py` is the formal test-only evaluation. It reports AutoAttack,
clean accuracy, four white-box attacks, and three black-box attacks, then writes
JSON and prints `EVAL_COMPLETE` only when every metric succeeds. Formal
stochastic attacks explicitly use seed 0, and evaluation rejects a mismatched
robust-teacher hash.

`setup_models.sh` only verifies existing shared checkpoints. The old
WRN-34-20 converter and generic teacher trainer are deliberately disabled to
prevent accidental replacement of public project weights.

This is a repaired source snapshot, not a prepared experiment directory. Its
bundled Slurm templates contain the integrity guards above, but retain
historical local paths/resources and are not direct submission commands.

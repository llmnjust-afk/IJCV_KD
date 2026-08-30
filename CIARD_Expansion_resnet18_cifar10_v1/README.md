# CIFAR-10 / ResNet-18 source

Fixed identity:

- student: ResNet-18
- robust teacher: raw WRN-34-10
- natural teacher: raw ResNet-56
- shared checkpoints: `models/model_cifar_wrn.pt` and
  `models/nat_teacher_checkpoint/cifar10_resnnet56.pth`

The ResNet line now implements the same SAA/RCD switches, temperature handling,
teacher warm-up semantics, strict loading, validation split, and result
integrity checks as the MobileNet line. It deliberately retains the existing
ResNet-specific base parameters, PCGrad teacher-margin path, and weight-space
averaging path.

`CIARD.py` exposes `--sard_saa`, `--sard_rcd`, `--epochs`, `--prefix`, and
`--original_ciard`. Checkpoint selection and weight-averaging inspection use the
fixed validation split; the CIFAR-10 test set is reserved for `attack_eval.py`.

`attack_eval.py` is the formal test-only evaluation. It reports AutoAttack,
clean accuracy, four white-box attacks, and three black-box attacks, then writes
JSON and prints `EVAL_COMPLETE` only when every metric succeeds.

This is a repaired source snapshot, not a prepared experiment directory. The
eight-run parameter matrix and new Slurm scripts will be created separately
after the experiment configurations are chosen.

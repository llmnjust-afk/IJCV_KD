# IJCV_KD 0830v1 source

This snapshot contains the SARD extension of CIARD for CIFAR-10 with two
independent student lines:

- `CIARD_Expansion_mobilenetv2_cifar10_v1`: MobileNet-V2 student.
- `CIARD_Expansion_resnet18_cifar10_v1`: ResNet-18 student, retaining its
  architecture-specific PCGrad and weight-averaging implementation.

Both lines now share the same SAA/RCD implementation and common correctness
fixes. In accordance with the CIARD paper, both use the existing raw
WRN-34-10 robust teacher and raw ResNet-56 natural teacher on CIFAR-10. The
student architectures remain different.

The latest local hardening also converts the epoch-51 teacher-update predicate
to a native Python `bool`, fixes formal stochastic evaluation to explicit seed
0, verifies the robust-teacher hash before attacks, and adds fail-closed
CUDA/data/teacher preflight plus `TRAIN_COMPLETE` checkpoint proof to the
bundled training templates. These changes do not alter model parameters,
losses, schedules, or attack budgets.

The verified local runtime is the existing `ciard` environment (Python 3.8,
PyTorch 1.10.0+cu113, torchvision 0.11.1+cu113). Teacher weights and CIFAR-10
data are shared from the project root through links in downstream run
directories; this source snapshot does not download or overwrite them.

This repository remains a source snapshot and contains no formal run outputs or
post-hardening performance results. Experiment-specific parameter matrices,
prefixes, resources, logs, and checkpoints are maintained in downstream
independent run directories rather than copied back here. See
`EXPERIMENT_GUIDE.md` for the implementation and evaluation interfaces.

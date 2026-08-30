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

The verified local runtime is the existing `ciard` environment (Python 3.8,
PyTorch 1.10.0+cu113, torchvision 0.11.1+cu113). Teacher weights and CIFAR-10
data are shared from the project root through links created in future run
directories; this source snapshot does not download or overwrite them.

No formal training/evaluation run directories or parameter matrix have been
prepared from this repaired source yet, and no cluster job has been submitted.
See `EXPERIMENT_GUIDE.md` for the implementation and evaluation interfaces.

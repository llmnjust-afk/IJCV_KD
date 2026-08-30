#!/bin/bash
#SBATCH --nodelist=aias-compute-4
#SBATCH --gres=gpu:1
#SBATCH --job-name=tr624_m10rep
#SBATCH --output=/home/lixidong25/mycode/CIARD_Expansion/CIARD_Expansion_mobilenetv2_cifar10_v1/logs/slurm/train_%j.out
#SBATCH --error=/home/lixidong25/mycode/CIARD_Expansion/CIARD_Expansion_mobilenetv2_cifar10_v1/logs/slurm/train_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
cd /home/lixidong25/mycode/CIARD_Expansion/CIARD_Expansion_mobilenetv2_cifar10_v1
module load cuda/11.8
source /apps/miniconda3/etc/profile.d/conda.sh
conda activate ciard
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
echo '========== CIARD 0624 TRAIN =========='
echo 'resolved training parameters are emitted by CIARD.py'
echo "data_path=$(readlink -f data)"
echo "models_path=$(readlink -f models)"
echo "allocated_cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo '================================================'

ROBUST_TEACHER='models/model_cifar_wrn.pt'
NATURAL_TEACHER='models/nat_teacher_checkpoint/cifar10_resnnet56.pth'
EXPECTED_ROBUST_TEACHER_SHA256='2ede52bd042bbdf40a0c27e8008034afd9cbb0b256b9077a255e555d25f957f4'
EXPECTED_NATURAL_TEACHER_SHA256='9e1d3395f0a8c34296ca8cd4875b9b5177d53f79e89af9b88e1a6724c6d6c860'

test -s "$ROBUST_TEACHER" || { echo "Missing robust teacher: $ROBUST_TEACHER" >&2; exit 1; }
test -s "$NATURAL_TEACHER" || { echo "Missing natural teacher: $NATURAL_TEACHER" >&2; exit 1; }
ROBUST_TEACHER_SHA256="$(sha256sum "$ROBUST_TEACHER" | awk '{print $1}')"
NATURAL_TEACHER_SHA256="$(sha256sum "$NATURAL_TEACHER" | awk '{print $1}')"
[[ "$ROBUST_TEACHER_SHA256" == "$EXPECTED_ROBUST_TEACHER_SHA256" ]] || {
    echo "Robust teacher SHA256 mismatch: expected $EXPECTED_ROBUST_TEACHER_SHA256, got $ROBUST_TEACHER_SHA256" >&2
    exit 1
}
[[ "$NATURAL_TEACHER_SHA256" == "$EXPECTED_NATURAL_TEACHER_SHA256" ]] || {
    echo "Natural teacher SHA256 mismatch: expected $EXPECTED_NATURAL_TEACHER_SHA256, got $NATURAL_TEACHER_SHA256" >&2
    exit 1
}
echo "robust_teacher_path=$(readlink -f "$ROBUST_TEACHER")"
echo "robust_teacher_sha256=$ROBUST_TEACHER_SHA256"
echo "natural_teacher_path=$(readlink -f "$NATURAL_TEACHER")"
echo "natural_teacher_sha256=$NATURAL_TEACHER_SHA256"

python - <<'PY'
import torch
import torchvision

if not torch.cuda.is_available():
    raise RuntimeError("Training preflight requires a CUDA GPU")
visible_gpu_count = torch.cuda.device_count()
if visible_gpu_count != 1:
    raise RuntimeError(
        "Training preflight requires exactly one visible GPU, got {}".format(
            visible_gpu_count))
probe = torch.empty(1, device="cuda:0")
properties = torch.cuda.get_device_properties(0)
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=False)
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=False)
if len(trainset) != 50000 or len(testset) != 10000:
    raise RuntimeError(
        "Unexpected CIFAR-10 sizes: train={}, test={}".format(
            len(trainset), len(testset)))
print("preflight_visible_gpu_count={}".format(visible_gpu_count))
print("preflight_gpu_name={}".format(properties.name))
print("preflight_gpu_total_memory_bytes={}".format(properties.total_memory))
print("preflight_dataset_samples=train:{} test:{}".format(
    len(trainset), len(testset)))
del probe
PY
echo 'PRECHECK_OK'

TRAIN_LOG="logs/train_stdout_${SLURM_JOB_ID}.log"
TRAIN_CKPT='model/Cifar10_MobileNetV2_tm010_repeat0620/student_best.pth'
python -u CIARD.py 2>&1 | tee "$TRAIN_LOG"
test -s "$TRAIN_CKPT" || {
    echo "Training exited without a non-empty checkpoint: $TRAIN_CKPT" >&2
    exit 1
}
TRAIN_CKPT_SHA256="$(sha256sum "$TRAIN_CKPT" | awk '{print $1}')"
TRAIN_CKPT_REALPATH="$(readlink -f "$TRAIN_CKPT")"
TRAIN_COMPLETE_LINE="TRAIN_COMPLETE checkpoint=$TRAIN_CKPT_REALPATH checkpoint_sha256=$TRAIN_CKPT_SHA256"
echo "$TRAIN_COMPLETE_LINE" | tee -a "$TRAIN_LOG"

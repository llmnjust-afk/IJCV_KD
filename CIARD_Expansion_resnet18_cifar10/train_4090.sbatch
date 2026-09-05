#!/bin/bash
#SBATCH --partition=rtx4090
#SBATCH --nodelist=aias-compute-2
#SBATCH --gres=gpu:4090:1
#SBATCH --job-name=tr96git
#SBATCH --output=/home/lixidong25/mycode/CIARD_Expansion/origin_code/0906v1/IJCV_KD/CIARD_Expansion_resnet18_cifar10/logs/slurm/train_%j.out
#SBATCH --error=/home/lixidong25/mycode/CIARD_Expansion/origin_code/0906v1/IJCV_KD/CIARD_Expansion_resnet18_cifar10/logs/slurm/train_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
cd /home/lixidong25/mycode/CIARD_Expansion/origin_code/0906v1/IJCV_KD/CIARD_Expansion_resnet18_cifar10
module load cuda/11.8
source /apps/miniconda3/etc/profile.d/conda.sh
conda activate ciard
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
unset CIARD_GPU CIARD_STUDENT CIARD_PREFIX CIARD_TM_WEIGHT CIARD_TM_START CIARD_TM_WARMUP
unset CIARD_TM_CLEAN_GATE CIARD_TM_ADV_GATE CIARD_TM_CONFLICT_FLOOR CIARD_TM_PS_CONFLICT CIARD_TM_PS_FLOOR CIARD_TM_RELATIVE
unset CIARD_TM_RELATIVE_ETA CIARD_TM_CLEAN_TAU CIARD_TM_ADV_TAU CIARD_TM_TWO_STAGE CIARD_TM_FINETUNE_START CIARD_TM_DETACHED

VARIANT='resnet18_tmix_a020_s120_w40_p081740'
TRAIN_CKPT='model/Cifar10_ResNet18_0906v1_tmix_a020_s120_w40_p081740/student_best.pth'
ROBUST_TEACHER='models/model_cifar_wrn.pt'
NATURAL_TEACHER='models/nat_teacher_checkpoint/cifar10_resnnet56.pth'
EXPECTED_ROBUST_SHA256='2ede52bd042bbdf40a0c27e8008034afd9cbb0b256b9077a255e555d25f957f4'
EXPECTED_NATURAL_SHA256='9e1d3395f0a8c34296ca8cd4875b9b5177d53f79e89af9b88e1a6724c6d6c860'

test ! -e "${TRAIN_CKPT%/*}" || { echo "Refusing existing training prefix: ${TRAIN_CKPT%/*}" >&2; exit 1; }
TRAIN_LOG="logs/train_stdout_${SLURM_JOB_ID}.log"
{
echo "========== 0906v1 TRAIN: $VARIANT =========="
echo "source_ciard_sha256=$(sha256sum CIARD.py | awk '{print $1}')"
echo "data_path=$(readlink -f data)"
echo "models_path=$(readlink -f models)"
echo "requested_node=aias-compute-2"
echo "running_node=${SLURMD_NODENAME:-unset}"
echo 'requested_gpu=1x4090'
echo "allocated_cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

test -s "$ROBUST_TEACHER" || { echo "Missing robust teacher: $ROBUST_TEACHER" >&2; exit 1; }
test -s "$NATURAL_TEACHER" || { echo "Missing natural teacher: $NATURAL_TEACHER" >&2; exit 1; }
ROBUST_SHA256="$(sha256sum "$ROBUST_TEACHER" | awk '{print $1}')"
NATURAL_SHA256="$(sha256sum "$NATURAL_TEACHER" | awk '{print $1}')"
[[ "$ROBUST_SHA256" == "$EXPECTED_ROBUST_SHA256" ]] || { echo "Robust teacher SHA256 mismatch: $ROBUST_SHA256" >&2; exit 1; }
[[ "$NATURAL_SHA256" == "$EXPECTED_NATURAL_SHA256" ]] || { echo "Natural teacher SHA256 mismatch: $NATURAL_SHA256" >&2; exit 1; }
echo "robust_teacher_sha256=$ROBUST_SHA256"
echo "natural_teacher_sha256=$NATURAL_SHA256"

python - <<'PY'
import torch
import torchvision
import loguru, numpy, torchattacks, autoattack
if torch.__version__ != '1.10.0+cu113' or torchvision.__version__ != '0.11.1+cu113':
    raise RuntimeError("Unexpected torch/torchvision environment")
if '4090' not in torch.cuda.get_device_name(0):
    raise RuntimeError("Expected the requested 4090 GPU type")
print("preflight_torch={} torchvision={} cuda_build={}".format(torch.__version__, torchvision.__version__, torch.version.cuda))

if not torch.cuda.is_available():
    raise RuntimeError("Training preflight requires CUDA")
if torch.cuda.device_count() != 1:
    raise RuntimeError("Expected exactly one visible GPU, got {}".format(torch.cuda.device_count()))
probe = torch.empty(1, device="cuda:0")
props = torch.cuda.get_device_properties(0)
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=False)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=False)
if len(trainset) != 50000 or len(testset) != 10000:
    raise RuntimeError("Unexpected CIFAR-10 sizes: train={} test={}".format(len(trainset), len(testset)))
print("preflight_gpu_name={}".format(props.name))
print("preflight_dataset_samples=train:{} test:{}".format(len(trainset), len(testset)))
del probe
PY
echo 'PRECHECK_OK'
} 2>&1 | tee "$TRAIN_LOG"

python -u CIARD.py 2>&1 | tee -a "$TRAIN_LOG"
test -s "$TRAIN_CKPT" || { echo "Training exited without checkpoint: $TRAIN_CKPT" >&2; exit 1; }
TRAIN_CKPT_SHA256="$(sha256sum "$TRAIN_CKPT" | awk '{print $1}')"
TRAIN_COMPLETE_LINE="TRAIN_COMPLETE variant=$VARIANT checkpoint=$(readlink -f "$TRAIN_CKPT") checkpoint_sha256=$TRAIN_CKPT_SHA256"
echo "$TRAIN_COMPLETE_LINE" | tee -a "$TRAIN_LOG"

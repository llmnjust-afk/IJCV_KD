#!/bin/bash
#SBATCH --partition=rtx4090
#SBATCH --nodelist=aias-compute-2
#SBATCH --gres=gpu:4090:1
#SBATCH --job-name=ev96git
#SBATCH --output=/home/lixidong25/mycode/CIARD_Expansion/origin_code/0906v1/IJCV_KD/CIARD_Expansion_resnet18_cifar10/logs/slurm/eval_best_%j.out
#SBATCH --error=/home/lixidong25/mycode/CIARD_Expansion/origin_code/0906v1/IJCV_KD/CIARD_Expansion_resnet18_cifar10/logs/slurm/eval_best_%j.err
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
CKPT='model/Cifar10_ResNet18_0906v1_tmix_a020_s120_w40_p081740/student_best.pth'
ROBUST_TEACHER='models/model_cifar_wrn.pt'
EXPECTED_ROBUST_SHA256='2ede52bd042bbdf40a0c27e8008034afd9cbb0b256b9077a255e555d25f957f4'

test -s "$CKPT" || { echo "Missing or empty checkpoint: $CKPT" >&2; exit 1; }
test -s "$ROBUST_TEACHER" || { echo "Missing robust teacher: $ROBUST_TEACHER" >&2; exit 1; }
ROBUST_SHA256="$(sha256sum "$ROBUST_TEACHER" | awk '{print $1}')"
[[ "$ROBUST_SHA256" == "$EXPECTED_ROBUST_SHA256" ]] || { echo "Robust teacher SHA256 mismatch: $ROBUST_SHA256" >&2; exit 1; }
CKPT_SHA256="$(sha256sum "$CKPT" | awk '{print $1}')"
EVALUATOR_SHA256="$(sha256sum attack_eval.py | awk '{print $1}')"

EVAL_LOG="logs/eval_best_stdout_${SLURM_JOB_ID}.log"
{
echo "========== 0906v1 EVAL BEST: $VARIANT =========="
echo "checkpoint=$(readlink -f "$CKPT")"
echo "checkpoint_sha256=$CKPT_SHA256"
echo "evaluator_sha256=$EVALUATOR_SHA256"
echo "robust_teacher_sha256=$ROBUST_SHA256"
echo 'evaluator_protocol=frozen_best_backup_historical; stochastic attack seeds are not explicitly fixed'
echo "requested_node=aias-compute-2"
echo "running_node=${SLURMD_NODENAME:-unset}"
echo 'requested_gpu=1x4090'
echo "allocated_cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

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
    raise RuntimeError("Evaluation preflight requires CUDA")
if torch.cuda.device_count() != 1:
    raise RuntimeError("Expected exactly one visible GPU, got {}".format(torch.cuda.device_count()))
probe = torch.empty(1, device="cuda:0")
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=False)
if len(testset) != 10000:
    raise RuntimeError("Unexpected CIFAR-10 test size: {}".format(len(testset)))
print("preflight_gpu_name={}".format(torch.cuda.get_device_name(0)))
print("preflight_test_samples={}".format(len(testset)))
PY
echo 'PRECHECK_OK'
} 2>&1 | tee "$EVAL_LOG"

python -u attack_eval.py 2>&1 | tee -a "$EVAL_LOG"
python -u check_eval_log.py "$EVAL_LOG" "$CKPT" "$CKPT_SHA256" "$EVALUATOR_SHA256" "$VARIANT" "$SLURM_JOB_ID" 2>&1 | tee -a "$EVAL_LOG"
EVAL_COMPLETE_LINE="EVAL_COMPLETE variant=$VARIANT checkpoint_sha256=$CKPT_SHA256 evaluator_sha256=$EVALUATOR_SHA256"
echo "$EVAL_COMPLETE_LINE" | tee -a "$EVAL_LOG"

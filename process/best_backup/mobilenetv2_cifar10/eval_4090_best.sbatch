#!/bin/bash
#SBATCH --nodelist=aias-compute-4
#SBATCH --gres=gpu:1
#SBATCH --job-name=ev624_m10rep
#SBATCH --output=/home/lixidong25/mycode/CIARD_Expansion/best_backup/mobilenetv2_cifar10/logs/slurm/eval_best_%j.out
#SBATCH --error=/home/lixidong25/mycode/CIARD_Expansion/best_backup/mobilenetv2_cifar10/logs/slurm/eval_best_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
cd /home/lixidong25/mycode/CIARD_Expansion/best_backup/mobilenetv2_cifar10
module load cuda/11.8
source /apps/miniconda3/etc/profile.d/conda.sh
conda activate ciard
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

CKPT='model/Cifar10_MobileNetV2_tm010_repeat0620/student_best.pth'
test -f "$CKPT" || { echo "Missing checkpoint: $CKPT" >&2; exit 1; }
echo '========== CIARD 0624 EVALUATION =========='
echo 'resolved evaluation parameters are emitted by attack_eval.py'
echo "data_path=$(readlink -f data)"
echo "models_path=$(readlink -f models)"
echo "allocated_cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo '==============================================='

python -u attack_eval.py 2>&1 | tee "logs/eval_best_stdout_${SLURM_JOB_ID}.log"

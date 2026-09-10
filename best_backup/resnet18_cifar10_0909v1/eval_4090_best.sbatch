#!/bin/bash
#SBATCH --partition=rtx4090
#SBATCH --nodelist=aias-compute-4
#SBATCH --gres=gpu:4090:1
#SBATCH --job-name=ev99br5
#SBATCH --output=/home/lixidong25/mycode/CIARD_Expansion/origin_code/0909v1/IJCV_KD/best_backup/resnet18_cifar10_0909v1/logs/slurm/eval_best_%j.out
#SBATCH --error=/home/lixidong25/mycode/CIARD_Expansion/origin_code/0909v1/IJCV_KD/best_backup/resnet18_cifar10_0909v1/logs/slurm/eval_best_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
cd /home/lixidong25/mycode/CIARD_Expansion/origin_code/0909v1/IJCV_KD/best_backup/resnet18_cifar10_0909v1
module load cuda/11.8
source /apps/miniconda3/etc/profile.d/conda.sh
conda activate ciard
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
unset CIARD_GPU CIARD_STUDENT CIARD_PREFIX
exec 9>logs/.eval_best.lock
flock -n 9 || { echo 'Another job for this experiment is running' >&2; exit 1; }
LOG="logs/eval_best_stdout_${SLURM_JOB_ID}.log"
test ! -e "$LOG" || { echo "Refusing existing log: $LOG" >&2; exit 1; }
{
echo 'requested_node=aias-compute-4 requested_gpu=1x4090'
echo "running_node=${SLURMD_NODENAME:-unset} allocated_cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
python -u run_checks.py verify
python -u run_checks.py preflight --cuda
} 2>&1 | tee "$LOG"
python -u attack_eval.py 2>&1 | tee -a "$LOG"
python -u check_eval_log.py "$LOG" "$SLURM_JOB_ID" 2>&1 | tee -a "$LOG"

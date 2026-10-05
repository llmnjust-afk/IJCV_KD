#!/bin/bash
#SBATCH --partition=a800
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=bak1004h2_tr
#SBATCH --output=/mnt/beegfs/home/lixidong25/mycode/CIARD_Expansion/origin_code/1004/IJCV_KD/best_backup/resnet18_cifar100_1004_h2/logs/slurm/train_%j.out
#SBATCH --error=/mnt/beegfs/home/lixidong25/mycode/CIARD_Expansion/origin_code/1004/IJCV_KD/best_backup/resnet18_cifar100_1004_h2/logs/slurm/train_%j.err

set -euo pipefail
cd /mnt/beegfs/home/lixidong25/mycode/CIARD_Expansion/origin_code/1004/IJCV_KD/best_backup/resnet18_cifar100_1004_h2
CIARD_PYTHON=/home/lixidong25/.conda/envs/ciard/bin/python
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
unset CIARD_GPU CIARD_STUDENT CIARD_PREFIX
while IFS= read -r CIARD_LEGACY; do unset "$CIARD_LEGACY"; done < <(compgen -A variable CIARD_TM_ || true)
exec 9>logs/.artifact.lock
flock -n 9 || { echo 'Another job for this group is running' >&2; exit 1; }
LOG="logs/train_stdout_${SLURM_JOB_ID}.log"
test ! -e "$LOG" || { echo "Refusing existing log: $LOG" >&2; exit 1; }
{
echo 'requested_partition=a800 requested_qos=urgent requested_gpu=1xA800'
echo "running_node=${SLURMD_NODENAME:-unset} allocated_cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
"$CIARD_PYTHON" -u run_checks.py start
"$CIARD_PYTHON" -u run_checks.py preflight --phase=train --cuda --expected-gpus=1
"$CIARD_PYTHON" -u check_a800_runtime.py --expected-gpus=1
} 2>&1 | tee "$LOG"
"$CIARD_PYTHON" -u CIARD.py 2>&1 | tee -a "$LOG"
"$CIARD_PYTHON" -u run_checks.py finish --log "$LOG" --job "$SLURM_JOB_ID" 2>&1 | tee -a "$LOG"

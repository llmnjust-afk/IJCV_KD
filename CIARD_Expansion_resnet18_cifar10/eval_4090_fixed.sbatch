#!/bin/bash
#SBATCH --partition=rtx4090
#SBATCH --nodelist=aias-compute-4
#SBATCH --gres=gpu:4090:1
#SBATCH --job-name=ev1007awpc_s1
#SBATCH --output=/home/lixidong25/mycode/CIARD_Expansion/run/1007-cifar10-r4-awpschedule-v1/resnet18_cifar10_c_s1/logs/slurm/eval_%j.out
#SBATCH --error=/home/lixidong25/mycode/CIARD_Expansion/run/1007-cifar10-r4-awpschedule-v1/resnet18_cifar10_c_s1/logs/slurm/eval_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
cd /home/lixidong25/mycode/CIARD_Expansion/run/1007-cifar10-r4-awpschedule-v1/resnet18_cifar10_c_s1
module load cuda/11.8
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
unset CIARD_GPU CIARD_STUDENT CIARD_PREFIX
exec 9>logs/.eval.lock
flock -n 9 || { echo 'Another job for this experiment is running' >&2; exit 1; }
LOG="logs/eval_stdout_${SLURM_JOB_ID}.log"
test ! -e "$LOG" || { echo "Refusing existing log: $LOG" >&2; exit 1; }
{
echo 'requested_node=aias-compute-4 requested_gpu=1x4090'
echo "running_node=${SLURMD_NODENAME:-unset} allocated_cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
/home/lixidong25/.conda/envs/ciard/bin/python -u run_checks.py verify
/home/lixidong25/.conda/envs/ciard/bin/python -u run_checks.py preflight --phase eval --cuda
} 2>&1 | tee "$LOG"
/home/lixidong25/.conda/envs/ciard/bin/python -u attack_eval.py 2>&1 | tee -a "$LOG"
/home/lixidong25/.conda/envs/ciard/bin/python -u check_eval_log.py "$LOG" "$SLURM_JOB_ID" 2>&1 | tee -a "$LOG"

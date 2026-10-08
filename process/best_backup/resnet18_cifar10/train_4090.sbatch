#!/bin/bash
#SBATCH --nodelist=aias-compute-4
#SBATCH --gres=gpu:1
#SBATCH --job-name=tr703_08
#SBATCH --output=/home/lixidong25/mycode/CIARD_Expansion/best_backup/resnet18_cifar10/logs/slurm/train_%j.out
#SBATCH --error=/home/lixidong25/mycode/CIARD_Expansion/best_backup/resnet18_cifar10/logs/slurm/train_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
cd /home/lixidong25/mycode/CIARD_Expansion/best_backup/resnet18_cifar10
module load cuda/11.8
source /apps/miniconda3/etc/profile.d/conda.sh
conda activate ciard
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
unset CIARD_GPU CIARD_STUDENT CIARD_PREFIX CIARD_TM_WEIGHT CIARD_TM_START CIARD_TM_WARMUP
unset CIARD_TM_CLEAN_GATE CIARD_TM_ADV_GATE CIARD_TM_CONFLICT_FLOOR CIARD_TM_PS_CONFLICT CIARD_TM_PS_FLOOR CIARD_TM_RELATIVE
unset CIARD_TM_RELATIVE_ETA CIARD_TM_CLEAN_TAU CIARD_TM_ADV_TAU CIARD_TM_TWO_STAGE CIARD_TM_FINETUNE_START CIARD_TM_DETACHED

echo '========== CIARD 0703 PCGRAD TRAIN: r18_pcgrad_optuna_transfer =========='
echo 'setting: 0629 Optuna-like clean/black-box region transferred to PCGrad; start rounded to 120 for WA checkpoint cadence'
echo 'resolved training parameters are emitted by CIARD.py'
echo "data_path=$(readlink -f data)"
echo "models_path=$(readlink -f models)"
echo "allocated_cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
python -u CIARD.py 2>&1 | tee "logs/train_stdout_${SLURM_JOB_ID}.log"

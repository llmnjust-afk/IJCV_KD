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

python -u CIARD.py 2>&1 | tee "logs/train_stdout_${SLURM_JOB_ID}.log"

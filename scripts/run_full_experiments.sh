#!/bin/bash
# run_full_experiments.sh — Run full SARD vs CIARD comparison experiments
#
# Exp 2: 100 epochs (SARD on GPU 0, CIARD baseline on GPU 1)
# Exp 3: 200 epochs (SARD on GPU 0, CIARD baseline on GPU 1)
#
# Usage: bash scripts/run_full_experiments.sh [EPOCHS]
# Default EPOCHS=200

set -euo pipefail

EPOCHS=${1:-200}
CODE_DIR="/data/lab/IJCV_KD/CIARD_Expansion_mobilenetv2_cifar10_v1"
LOG_DIR="${CODE_DIR}/logs"
mkdir -p "$LOG_DIR"

cd "$CODE_DIR"

echo "============================================"
echo "Full SARD vs CIARD Comparison: ${EPOCHS} epochs"
echo "Start time: $(date)"
echo "============================================"

# Run SARD and CIARD baseline in parallel on dual GPUs
echo "[GPU 0] Starting SARD (${EPOCHS} epochs)..."
CIARD_GPU=0 python3 CIARD.py --sard_saa 1 --sard_rcd 1 --epochs $EPOCHS --prefix sard_full > "${LOG_DIR}/sard_full_${EPOCHS}ep.log" 2>&1 &
PID_SARD=$!
echo "  PID: $PID_SARD"

echo "[GPU 1] Starting CIARD baseline (${EPOCHS} epochs)..."
CIARD_GPU=1 python3 CIARD.py --sard_saa 0 --sard_rcd 0 --epochs $EPOCHS --prefix baseline_full > "${LOG_DIR}/baseline_full_${EPOCHS}ep.log" 2>&1 &
PID_BASE=$!
echo "  PID: $PID_BASE"

echo ""
echo "Both experiments running in parallel..."
echo "Monitor: tail -f ${LOG_DIR}/sard_full_${EPOCHS}ep.log ${LOG_DIR}/baseline_full_${EPOCHS}ep.log"
echo ""

wait $PID_SARD
echo "[GPU 0] SARD completed at $(date)"

wait $PID_BASE
echo "[GPU 1] CIARD baseline completed at $(date)"

echo ""
echo "============================================"
echo "Full experiments completed!"
echo "End time: $(date)"
echo "============================================"

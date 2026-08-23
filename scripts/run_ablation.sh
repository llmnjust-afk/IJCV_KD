#!/bin/bash
# run_ablation.sh — Run SARD ablation experiments (60 epochs, 4 configs) on dual GPUs
#
# GPU 0: baseline (no SAA, no RCD) → SAA only
# GPU 1: RCD only → SARD full (SAA + RCD)
#
# Usage: bash scripts/run_ablation.sh [EPOCHS]
# Default EPOCHS=60

set -euo pipefail

EPOCHS=${1:-60}
CODE_DIR="/data/lab/IJCV_KD/CIARD_Expansion_mobilenetv2_cifar10_v1"
LOG_DIR="${CODE_DIR}/logs"
mkdir -p "$LOG_DIR"

cd "$CODE_DIR"

echo "============================================"
echo "SARD Ablation Study: ${EPOCHS} epochs, 4 configs on 2 GPUs"
echo "Start time: $(date)"
echo "============================================"

# === GPU 0: Config 1 — Baseline (no SAA, no RCD) ===
echo "[GPU 0] Starting baseline (no SAA, no RCD)..."
CIARD_GPU=0 python3 CIARD.py --sard_saa 0 --sard_rcd 0 --epochs $EPOCHS --prefix ablation_baseline > "${LOG_DIR}/ablation_baseline.log" 2>&1 &
PID_BASELINE=$!
echo "  PID: $PID_BASELINE, Log: ablation_baseline.log"

# === GPU 1: Config 3 — RCD only ===
echo "[GPU 1] Starting RCD only..."
CIARD_GPU=1 python3 CIARD.py --sard_saa 0 --sard_rcd 1 --epochs $EPOCHS --prefix ablation_rcd_only > "${LOG_DIR}/ablation_rcd_only.log" 2>&1 &
PID_RCD=$!
echo "  PID: $PID_RCD, Log: ablation_rcd_only.log"

echo ""
echo "Waiting for first pair to complete..."
wait $PID_BASELINE
echo "[GPU 0] Baseline completed at $(date)"

wait $PID_RCD
echo "[GPU 1] RCD only completed at $(date)"

# === GPU 0: Config 2 — SAA only ===
echo ""
echo "[GPU 0] Starting SAA only..."
CIARD_GPU=0 python3 CIARD.py --sard_saa 1 --sard_rcd 0 --epochs $EPOCHS --prefix ablation_saa_only > "${LOG_DIR}/ablation_saa_only.log" 2>&1 &
PID_SAA=$!
echo "  PID: $PID_SAA, Log: ablation_saa_only.log"

# === GPU 1: Config 4 — SARD full (SAA + RCD) ===
echo "[GPU 1] Starting SARD full (SAA + RCD)..."
CIARD_GPU=1 python3 CIARD.py --sard_saa 1 --sard_rcd 1 --epochs $EPOCHS --prefix ablation_sard_full > "${LOG_DIR}/ablation_sard_full.log" 2>&1 &
PID_SARD=$!
echo "  PID: $PID_SARD, Log: ablation_sard_full.log"

echo ""
echo "Waiting for second pair to complete..."
wait $PID_SAA
echo "[GPU 0] SAA only completed at $(date)"

wait $PID_SARD
echo "[GPU 1] SARD full completed at $(date)"

echo ""
echo "============================================"
echo "All ablation experiments completed!"
echo "End time: $(date)"
echo "============================================"
echo ""
echo "Logs:"
ls -la "${LOG_DIR}"/ablation_*.log
echo ""
echo "Model checkpoints:"
ls -la "${CODE_DIR}"/model/ablation_*/student_best.pth 2>/dev/null || echo "No checkpoints found (check logs for errors)"

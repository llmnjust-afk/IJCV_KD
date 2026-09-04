#!/bin/bash
# eval_all.sh — Evaluate all trained student checkpoints with fast_eval.py
#
# Usage: bash scripts/eval_all.sh
# Run after experiments complete to evaluate all models.

set -euo pipefail

CODE_DIR="/data/lab/IJCV_KD/CIARD_Expansion_mobilenetv2_cifar10"
RESULTS_DIR="/data/lab/IJCV_KD/results"
mkdir -p "$RESULTS_DIR"

cd "$CODE_DIR"

echo "============================================"
echo "Evaluating all student checkpoints"
echo "Time: $(date)"
echo "============================================"

# Find all student_best.pth checkpoints
for ckpt in model/*/student_best.pth; do
    prefix=$(basename $(dirname "$ckpt"))
    echo ""
    echo "--- Evaluating: $prefix ---"
    result_file="${RESULTS_DIR}/${prefix}_eval.txt"
    
    CIARD_GPU=0 python3 fast_eval.py --checkpoint "$ckpt" --prefix "$prefix" 2>&1 | tee "$result_file"
    echo "Results saved to: $result_file"
done

echo ""
echo "============================================"
echo "All evaluations complete!"
echo "Time: $(date)"
echo "============================================"
echo ""
echo "Results directory:"
ls -la "$RESULTS_DIR"/*.txt 2>/dev/null
echo ""
echo "Summary:"
for f in "$RESULTS_DIR"/*.txt; do
    echo "--- $(basename $f) ---"
    grep -E "Clean|WB|BB|SUMMARY" "$f" 2>/dev/null
done

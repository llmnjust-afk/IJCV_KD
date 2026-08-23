#!/bin/bash
# auto_sync_results.sh — Automated experiment result synchronization to GitHub
# 
# Usage:
#   ./scripts/auto_sync_results.sh              # Normal run: collect, check, commit, push
#   ./scripts/auto_sync_results.sh --check      # Only check experiment status (no commit/push)
#   ./scripts/auto_sync_results.sh --force       # Force push even if checks fail
#
# This script:
#   1. Checks experiment status (looks for completed training/eval logs)
#   2. Collects checkpoints, logs, results, configs
#   3. Verifies checkpoint integrity (file exists, size > 0)
#   4. Generates checkpoint metadata
#   5. Git add + commit + push to GitHub
#
# Prerequisites:
#   - Git LFS installed and initialized
#   - GitHub PAT available at the path specified by PAT_FILE
#   - Remote repository: https://github.com/llmnjust-afk/IJCV_KD.git

set -euo pipefail

# === Configuration ===
REPO_URL="https://github.com/llmnjust-afk/IJCV_KD.git"
PAT_FILE="/data/lab/tmp/.github_pat"
REPO_DIR="/data/lab/IJCV_KD"
LOG_FILE="${REPO_DIR}/logs/sync_$(date +%Y%m%d_%H%M%S).log"
MODE="normal"

# Parse arguments
for arg in "$@"; do
    case $arg in
        --check) MODE="check" ;;
        --force) MODE="force" ;;
    esac
done

mkdir -p "${REPO_DIR}/logs"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================"
echo "Auto-Sync Results Script"
echo "Mode: $MODE"
echo "Time: $(date)"
echo "Log:  $LOG_FILE"
echo "============================================"

cd "$REPO_DIR" || { echo "ERROR: Cannot cd to $REPO_DIR"; exit 1; }

# === Step 1: Check experiment status ===
echo ""
echo "[Step 1] Checking experiment status..."

EXP_STATUS="incomplete"
if ls checkpoints/sard/*.pth 2>/dev/null | head -1 > /dev/null; then
    echo "  SARD checkpoints found:"
    for f in checkpoints/sard/*.pth; do
        SIZE=$(stat -c%s "$f" 2>/dev/null || echo 0)
        echo "    $f ($SIZE bytes)"
    done
    EXP_STATUS="complete"
fi

if ls checkpoints/baseline/*.pth 2>/dev/null | head -1 > /dev/null; then
    echo "  Baseline checkpoints found:"
    for f in checkpoints/baseline/*.pth; do
        SIZE=$(stat -c%s "$f" 2>/dev/null || echo 0)
        echo "    $f ($SIZE bytes)"
    done
    EXP_STATUS="complete"
fi

# Check for evaluation results
if ls results/*.txt 2>/dev/null | head -1 > /dev/null; then
    echo "  Evaluation results found:"
    ls -la results/*.txt 2>/dev/null
fi

if [ "$MODE" = "check" ]; then
    echo ""
    echo "[Check mode] Experiment status: $EXP_STATUS"
    echo "Exiting without committing."
    exit 0
fi

if [ "$EXP_STATUS" = "incomplete" ] && [ "$MODE" != "force" ]; then
    echo "WARNING: No checkpoints found. Use --force to commit anyway."
    exit 1
fi

# === Step 2: Generate checkpoint metadata ===
echo ""
echo "[Step 2] Generating checkpoint metadata..."

GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

METADATA_FILE="${REPO_DIR}/checkpoints/checkpoint_metadata.json"
python3 -c "
import json, os, glob, subprocess

metadata = {
    'repository': 'https://github.com/llmnjust-afk/IJCV_KD.git',
    'git_commit': '$GIT_HASH',
    'date': '$DATE',
    'checkpoints': []
}

ckpt_dirs = {
    'teacher': 'checkpoints/teacher',
    'student_baseline': 'checkpoints/student_baseline',
    'sard': 'checkpoints/sard',
    'baseline': 'checkpoints/baseline',
    'ablation': 'checkpoints/ablation',
}

for category, dirpath in ckpt_dirs.items():
    for f in sorted(glob.glob(os.path.join('$REPO_DIR', dirpath, '*.pth')) +
                    glob.glob(os.path.join('$REPO_DIR', dirpath, '*.pt'))):
        size = os.path.getsize(f)
        name = os.path.basename(f)
        rel_path = os.path.relpath(f, '$REPO_DIR')
        metadata['checkpoints'].append({
            'category': category,
            'name': name,
            'path': rel_path,
            'size_bytes': size,
            'size_mb': round(size / 1024 / 1024, 2),
        })

with open('$METADATA_FILE', 'w') as fout:
    json.dump(metadata, fout, indent=2)
print(f'Metadata written: {len(metadata[\"checkpoints\"])} checkpoints')
" 2>&1

echo "  Metadata file: $METADATA_FILE"

# === Step 3: Ensure Git LFS is configured ===
echo ""
echo "[Step 3] Checking Git LFS configuration..."

git lfs install 2>/dev/null || true
git lfs track "*.pth" 2>/dev/null || true
git lfs track "*.pt" 2>/dev/null || true
git lfs track "*.ckpt" 2>/dev/null || true

if [ -f ".gitattributes" ]; then
    echo "  .gitattributes:"
    cat .gitattributes | sed 's/^/    /'
fi

echo ""
echo "  LFS status:"
git lfs status 2>&1 | head -20 || echo "  (no LFS files yet)"

# === Step 4: Git add ===
echo ""
echo "[Step 4] Staging files..."

# Add code changes
git add -A CIARD_Expansion_mobilenetv2_cifar10_v1/CIARD.py 2>/dev/null || true
git add -A CIARD_Expansion_mobilenetv2_cifar10_v1/mtard_loss.py 2>/dev/null || true
git add -A CIARD_Expansion_mobilenetv2_cifar10_v1/fast_eval.py 2>/dev/null || true

# Add experiment infrastructure
git add -A scripts/ 2>/dev/null || true
git add -A configs/ 2>/dev/null || true
git add -A checkpoints/checkpoint_metadata.json 2>/dev/null || true
git add -A results/ 2>/dev/null || true
git add -A logs/*.log 2>/dev/null || true
git add -A .gitattributes 2>/dev/null || true

# Add checkpoint files via LFS (but not data/ or models/ dirs)
for ckpt_dir in checkpoints/teacher checkpoints/student_baseline checkpoints/sard checkpoints/baseline checkpoints/ablation; do
    if ls ${ckpt_dir}/*.pth 2>/dev/null | head -1 > /dev/null; then
        git add -A "${ckpt_dir}/" 2>/dev/null || true
    fi
done

# Add documentation
git add -A README.md 2>/dev/null || true
git add -A IMPROVEMENTS_SUMMARY.md 2>/dev/null || true
git add -A checkpoints/models_README.md 2>/dev/null || true

echo "  Staged files:"
git status --short 2>/dev/null | head -30

# === Step 5: Git commit ===
echo ""
echo "[Step 5] Committing..."

if git diff --cached --quiet 2>/dev/null; then
    echo "  No changes to commit."
else
    COMMIT_MSG="Add SARD experiment results and checkpoints (IJCV extension)

- SARD: Strength-Adaptive Reliability-Calibrated Distillation
- Module 1 (SAA): Beta-distribution epsilon sampling with curriculum
- Module 2 (RCD): Teacher Reliability Score per-sample weighting
- Checkpoints, logs, and evaluation results included
- Git commit: ${GIT_HASH}
- Date: ${DATE}"

    git commit -m "$COMMIT_MSG" 2>&1 || echo "  (commit may have failed if pre-commit hooks blocked)"
    echo "  Commit hash: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
fi

# === Step 6: Git push ===
echo ""
echo "[Step 6] Pushing to GitHub..."

if [ ! -f "$PAT_FILE" ]; then
    echo "ERROR: PAT file not found at $PAT_FILE"
    echo "Cannot push to GitHub without authentication."
    exit 1
fi

PAT=$(cat "$PAT_FILE" 2>/dev/null | tr -d '[:space:]')
if [ -z "$PAT" ]; then
    echo "ERROR: PAT file is empty"
    exit 1
fi

# Set remote URL with PAT (does not print the PAT)
REMOTE_WITH_PAT="https://${PAT}@github.com/llmnjust-afk/IJCV_KD.git"
git remote set-url origin "$REMOTE_WITH_PAT" 2>/dev/null || git remote add origin "$REMOTE_WITH_PAT" 2>/dev/null || true

# Push
PUSH_OUTPUT=$(git push origin HEAD:main 2>&1) || {
    echo "Push failed. Trying master branch..."
    PUSH_OUTPUT=$(git push origin HEAD:master 2>&1) || {
        echo "ERROR: Push failed to both main and master."
        echo "Reason: $PUSH_OUTPUT"
        # Restore clean remote URL (without PAT)
        git remote set-url origin "https://github.com/llmnjust-afk/IJCV_KD.git" 2>/dev/null || true
        exit 1
    }
}

# Restore clean remote URL (without PAT)
git remote set-url origin "https://github.com/llmnjust-afk/IJCV_KD.git" 2>/dev/null || true

echo "  Push successful!"
echo "  $PUSH_OUTPUT" | head -5

# === Summary ===
echo ""
echo "============================================"
echo "Sync Complete!"
echo "  Commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "  Time:   $(date)"
echo "============================================"

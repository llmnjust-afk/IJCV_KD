# IJCV_KD: SARD — Strength-Adaptive Reliability-Calibrated Distillation

IJCV journal extension of ICCV 2025 paper **CIARD** (Cyclic Iterative Adversarial Robustness Distillation).

## What's New (SARD)

SARD adds two modules to CIARD:
- **SAA** (Strength-Adaptive Attack): Beta-distribution epsilon sampling with curriculum
- **RCD** (Reliability-Calibrated Distillation): Per-sample Teacher Reliability Score weighting

## Quick Start

```bash
git clone https://github.com/llmnjust-afk/IJCV_KD.git
cd IJCV_KD/CIARD_Expansion_mobilenetv2_cifar10_v1
pip install torch torchvision loguru torchattacks autoattack robustbench
bash setup_models.sh                    # Download teacher models
python CIARD.py --sard_saa 1 --sard_rcd 1 --epochs 200 --prefix sard_200ep
python fast_eval.py --checkpoint model/sard_200ep/student_best.pth
```

## Documentation

- **[EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)** — Complete guide: setup, training, evaluation, ablation, code modifications
- **[CIARD_Expansion_mobilenetv2_cifar10_v1/README.md](CIARD_Expansion_mobilenetv2_cifar10_v1/README.md)** — Original CIARD README

## Directory Layout

| Directory | Description |
|-----------|-------------|
| `CIARD_Expansion_mobilenetv2_cifar10_v1/` | Main experiment (MobileNetV2 student, SARD-enabled) |
| `CIARD_Expansion_resnet18_cifar10_v1/` | ResNet18 variant (original CIARD, for reference) |
| `scripts/` | Experiment runner and analysis scripts |
| `configs/` | Experiment configurations |
| `results/` | Experiment results |

## Ablation Results (60 epochs)

| Metric | CIARD Baseline | SARD | Delta |
|--------|---------------|------|-------|
| Clean Acc | 89.63% | **92.42%** | +2.79% |
| Robust Acc | 1.95% | **3.14%** | +1.19% |
| Best Combined | 49.53% | **52.34%** | +2.81% |

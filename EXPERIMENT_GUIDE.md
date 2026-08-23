# SARD: Strength-Adaptive Reliability-Calibrated Distillation

IJCV journal extension of the ICCV 2025 paper **CIARD** (Cyclic Iterative Adversarial Robustness Distillation).

SARD introduces two modules on top of CIARD:
1. **SAA** (Strength-Adaptive Attack) — Beta-distribution epsilon sampling with curriculum, replacing fixed-epsilon adversarial example generation
2. **RCD** (Reliability-Calibrated Distillation) — Per-sample Teacher Reliability Score (TRS) weighting that down-weights distillation from unreliable teacher predictions

## Quick Start

```bash
# 1. Clone
git clone https://github.com/llmnjust-afk/IJCV_KD.git
cd IJCV_KD/CIARD_Expansion_mobilenetv2_cifar10_v1

# 2. Install dependencies
pip install torch torchvision loguru torchattacks autoattack robustbench

# 3. Download teacher models (CIFAR-10 + WRN-34-20 robust + ResNet-56 natural)
bash setup_models.sh

# 4. Train SARD (200 epochs)
python CIARD.py --sard_saa 1 --sard_rcd 1 --epochs 200 --prefix sard_200ep

# 5. Train CIARD baseline (200 epochs, run in parallel on another GPU)
python CIARD.py --sard_saa 0 --sard_rcd 0 --epochs 200 --prefix baseline_200ep

# 6. Evaluate
python fast_eval.py --checkpoint model/sard_200ep/student_best.pth --prefix sard_200ep
```

## GPU Selection

Set the `CIARD_GPU` environment variable before running:

```bash
CIARD_GPU=0 python CIARD.py --sard_saa 1 --sard_rcd 1 --epochs 200 --prefix sard_200ep
CIARD_GPU=1 python CIARD.py --sard_saa 0 --sard_rcd 0 --epochs 200 --prefix baseline_200ep
```

## Ablation Study

Run all four configurations to isolate SAA and RCD contributions:

| Config | SAA | RCD | Command |
|--------|-----|-----|---------|
| Baseline (CIARD) | 0 | 0 | `python CIARD.py --sard_saa 0 --sard_rcd 0 --epochs 60 --prefix ablation_baseline` |
| SAA only | 1 | 0 | `python CIARD.py --sard_saa 1 --sard_rcd 0 --epochs 60 --prefix ablation_saa_only` |
| RCD only | 0 | 1 | `python CIARD.py --sard_saa 0 --sard_rcd 1 --epochs 60 --prefix ablation_rcd_only` |
| SARD (SAA+RCD) | 1 | 1 | `python CIARD.py --sard_saa 1 --sard_rcd 1 --epochs 60 --prefix ablation_sard_full` |

## Evaluation

### Fast Evaluation (white-box + black-box, ~2 min)

```bash
python fast_eval.py --checkpoint model/sard_200ep/student_best.pth --prefix sard_200ep
```

Runs: Clean Accuracy, WB PGD-TRADES (20-step), WB PGD-SAT (20-step), WB FGSM, WB CW L-inf, BB PGD-TRADES, BB CW L-inf.

### Full Evaluation (includes AutoAttack, ~30 min)

Edit `attack_eval.py` to set the checkpoint path, then:

```bash
python attack_eval.py
```

Runs: Clean, PGD-20, FGSM, CW, AutoAttack (APGD-CE, APGD-DLR, FAB, Square).

## Teacher Models

SARD uses two teacher models:

| Teacher | Architecture | Source | Clean Acc | Role |
|---------|-------------|--------|-----------|------|
| Robust teacher | WRN-34-20 | RobustBench Rice2020Overfitting | ~85% | Adversarial distillation |
| Natural teacher | ResNet-56 | chenyaofo/pytorch-cifar-models | ~94% | Clean distillation |

`setup_models.sh` downloads and converts both automatically:

```bash
bash setup_models.sh
```

Manual setup (if `setup_models.sh` fails):

```bash
# Robust teacher: convert from RobustBench
python convert_rb_teacher.py
# → outputs models/model_cifar_wrn.pt (~736MB)

# Natural teacher: download from chenyaofo
wget -O models/nat_teacher_checkpoint/cifar10_resnet56_chenyaofo.pt \
  https://github.com/chenyaofo/pytorch-cifar-models/releases/download/v1.0/cifar10-resnet56-951c35a1.pth
# Then load with cifar10_resnet56(normalize=True) and re-save as cifar10_resnnet56.pth
```

## CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--sard_saa` | int | 1 | Enable SAA module (0=off, 1=on) |
| `--sard_rcd` | int | 1 | Enable RCD module (0=off, 1=on) |
| `--epochs` | int | 300 | Total training epochs |
| `--prefix` | str | `Cifar10_MobileNetV2_tm010_repeat0620` | Model save directory name |

## Checkpoint Output

Training saves checkpoints to `model/<prefix>/`:

| File | Save Condition | Content |
|------|----------------|---------|
| `student_<epoch>.pth` | Every `epochs//6` epochs | Student model + optimizer + epoch |
| `student_latest.pth` | Last 17% of epochs (every epoch) | Student model + optimizer + epoch |
| `student_best.pth` | When (clean+robust)/2 improves | Best student model |

Each checkpoint is a dict with keys: `model`, `optimizer`, `epoch`, and optionally `raw_student`, `ema_student`.

## Key Code Modifications vs Original CIARD

### 1. SARD SAA Module (`mtard_loss.py`)
- `sample_epsilon_curriculum(epoch, total_epochs)`: Samples perturbation epsilon from a Beta distribution with a curriculum that increases strength over training. Starts at small epsilon (~1/255) and grows to 8/255.

### 2. SARD RCD Module (`mtard_loss.py`)
- `teacher_reliability_score(teacher_logits, labels)`: Computes per-sample TRS based on teacher prediction confidence and margin. Down-weights KL distillation when the teacher is unreliable on adversarial inputs.

### 3. Teacher Model Fixes (`cifar10_models/wideresnet.py`)
- Added `widen_factor` parameter (default=10, set to 20 for Rice2020 WRN-34-20)
- Added `normalize` parameter with built-in CIFAR-10 mean/std normalization
- This fixes a critical bug: the RobustBench Rice2020 model internally normalizes inputs, but the original CIARD code fed raw [0,1] images

### 4. Natural Teacher Fix (`cifar10_nat_teacher_models/resnet.py`)
- Added `normalize` parameter to `CifarResNet.__init__` and `forward`
- Without normalization: 52% accuracy; with normalization: 94% accuracy

### 5. Training Hyperparameter Fixes (`CIARD.py`)
- `clean_ce_weight`: 0.05 → **0.3** (stronger CE learning signal)
- `student_ema_decay`: 0.999 → **0.995** (faster EMA convergence)
- `clean_ce_gate_floor`: 0.0 → **0.5** (minimum 50% CE signal through gate)
- Loss start epochs: changed from absolute epoch numbers to **fraction-of-training** scaling (so short experiments still get CE/margin/anchor losses early enough)
- Teacher checkpoint saves **disabled** (772MB each for WRN-34-20)

## Directory Structure

```
IJCV_KD/
├── CIARD_Expansion_mobilenetv2_cifar10_v1/   ← Main experiment (MobileNetV2 student)
│   ├── CIARD.py                ← Main training script (SARD integrated)
│   ├── mtard_loss.py           ← Loss functions + SARD modules (SAA, RCD)
│   ├── fast_eval.py            ← Quick evaluation (WB + BB attacks)
│   ├── attack_eval.py          ← Full evaluation (includes AutoAttack)
│   ├── convert_rb_teacher.py   ← Convert RobustBench WRN-34-20 to code format
│   ├── setup_models.sh         ← One-click teacher model download & conversion
│   ├── train_teacher.py        ← Train custom teacher (optional)
│   ├── validate_ciardpp.py     ← CIARD++ validation utilities
│   ├── requirements.txt        ← Python dependencies
│   ├── cifar10_models/         ← Student + robust teacher architectures
│   │   ├── wideresnet.py       ← WRN with widen_factor + normalize params
│   │   ├── mobilenet_v2.py     ← MobileNetV2 student
│   │   └── resnet.py           ← ResNet variants
│   ├── cifar10_nat_teacher_models/  ← Natural teacher architectures
│   │   ├── resnet.py           ← ResNet-56 with normalize param
│   │   └── ...
│   ├── models/                 ← Teacher checkpoints (gitignored, use setup_models.sh)
│   │   ├── model_cifar_wrn.pt              ← WRN-34-20 robust teacher (~736MB)
│   │   └── nat_teacher_checkpoint/         ← Natural teacher
│   ├── model/                  ← Training output checkpoints (gitignored)
│   │   └── <prefix>/
│   │       ├── student_best.pth
│   │       ├── student_latest.pth
│   │       └── student_<epoch>.pth
│   └── data/                   ← CIFAR-10 dataset (gitignored, auto-downloaded)
├── CIARD_Expansion_resnet18_cifar10_v1/      ← ResNet18 variant (original CIARD)
├── scripts/                    ← Experiment runner scripts
│   ├── run_ablation.sh
│   ├── run_full_experiments.sh
│   ├── eval_all.sh
│   ├── analyze_results.py
│   └── auto_sync_results.sh
├── configs/                    ← Experiment configurations
├── checkpoints/                ← Checkpoint documentation
├── results/                    ← Experiment results
├── .gitattributes              ← Git LFS for .pth/.pt/.ckpt files
└── .gitignore
```

## Ablation Results (60 epochs, preliminary)

| Metric | Baseline (CIARD) | SARD (SAA+RCD) | Delta |
|--------|-----------------|-----------------|-------|
| Final Clean Acc | 89.63% | **92.42%** | +2.79% |
| Final Robust Acc | 1.95% | **3.14%** | +1.19% |
| Best Clean Acc | 87.41% (ep24) | **92.82%** (ep52) | +5.41% |
| Best Robust Acc | 11.65% (ep24) | **13.85%** (ep44) | +2.20% |
| Best Combined | 49.53% | **52.34%** | +2.81% |

> Note: 60-epoch training has limited adversarial training time. The final 200-epoch experiment is expected to show significantly higher robust accuracy for both methods.

## Environment

- Python 3.8+
- PyTorch 2.0+ with CUDA
- Key packages: `torch`, `torchvision`, `loguru`, `torchattacks`, `autoattack`, `robustbench`

```bash
pip install torch torchvision loguru torchattacks autoattack robustbench
```

## SARD Method Details

### SAA: Strength-Adaptive Attack

Instead of using a fixed perturbation budget epsilon=8/255 for all adversarial example generation, SAA samples epsilon from a Beta(2, 5) distribution scaled to [1/255, 8/255]. The distribution shifts toward larger epsilon as training progresses (curriculum), allowing the student to gradually adapt to stronger adversarial perturbations.

**Implementation**: `sample_epsilon_curriculum(epoch, total_epochs, eps_max=8/255, eps_min=1/255)` in `mtard_loss.py`.

### RCD: Reliability-Calibrated Distillation

The robust teacher (WRN-34-20) achieves only ~58% robust accuracy under PGD-20, meaning ~42% of its adversarial predictions are wrong. RCD computes a per-sample Teacher Reliability Score (TRS) that measures how confident and correct the teacher is on each adversarial example. Samples where the teacher is unreliable receive lower weight in the KL distillation loss.

**Implementation**: `teacher_reliability_score(teacher_logits, labels, temperature, floor)` in `mtard_loss.py`.

The TRS is computed as:
1. Teacher prediction confidence (softmax probability of predicted class)
2. Margin between top-1 and top-2 logits
3. Normalized and clamped to [floor, 1.0] (floor=0.1 ensures minimum signal)

The TRS weight is applied to the KL divergence loss between teacher and student on adversarial examples.

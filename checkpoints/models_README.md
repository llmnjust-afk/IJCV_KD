# Model Checkpoints

This directory contains model checkpoints for the IJCV KD project. Due to GitHub file size limits, large checkpoints are managed via Git LFS or hosted externally.

## Teacher Models

### Robust Teacher: WideResNet-34-10
- **File**: `CIARD_Expansion_mobilenetv2_cifar10/models/model_cifar_wrn.pt`
- **Architecture**: WideResNet-34-10 (depth=34, widen_factor=10)
- **Training**: Standard CE (60 epochs) + PGD adversarial training (30 epochs)
- **Epsilon**: 8/255 Linf
- **Size**: ~190 MB
- **Usage**: `teacher = wideresnet(); teacher.load_state_dict(torch.load('models/model_cifar_wrn.pt'))`

### Natural Teacher: ResNet-56
- **File**: `CIARD_Expansion_mobilenetv2_cifar10/models/nat_teacher_checkpoint/cifar10_resnnet56.pth`
- **Source**: [chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models)
- **Architecture**: CifarResNet-56 (BasicBlock, [9,9,9])
- **Clean Accuracy**: ~93.18% on CIFAR-10
- **Size**: ~2.2 MB
- **Usage**: `teacher_nat = cifar10_resnet56(); teacher_nat.load_state_dict(torch.load('models/nat_teacher_checkpoint/cifar10_resnnet56.pth'))`

## Student Models

### SARD (Strength-Adaptive Reliability-Calibrated Distillation)
- **Location**: `checkpoints/sard/`
- **Architecture**: MobileNetV2
- **Training**: SAA + RCD modules enabled
- **Evaluation**: See `results/` directory

### CIARD Baseline
- **Location**: `checkpoints/baseline/`
- **Architecture**: MobileNetV2
- **Training**: Original CIARD configuration (fixed epsilon, equal-weight KL)

## Checkpoint Format

All student checkpoints are saved as:
```python
{
    'model': state_dict,           # student model weights
    'optimizer': optimizer_state,  # SGD optimizer state
    'epoch': int,                  # training epoch
    'raw_student': state_dict,     # raw (non-EMA) student (if EMA enabled)
    'ema_student': state_dict,     # EMA student (if EMA enabled)
}
```

## Reproduction

```bash
# 1. Clone repository
git clone https://github.com/llmnjust-afk/IJCV_KD.git
cd IJCV_KD

# 2. Install dependencies
pip install torch torchvision loguru torchattacks autoattack

# 3. Download teacher models (see above)

# 4. Train SARD
cd CIARD_Expansion_mobilenetv2_cifar10
python CIARD.py --sard_saa 1 --sard_rcd 1 --epochs 200 --prefix sard_final

# 5. Evaluate
python fast_eval.py --checkpoint model/sard_final/student_best.pth
```

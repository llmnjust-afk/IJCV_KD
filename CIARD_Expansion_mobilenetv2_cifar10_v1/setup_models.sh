#!/bin/bash
# setup_models.sh — Download/convert all required teacher models for SARD experiments
#
# Prerequisites:
#   pip install torch torchvision robustbench loguru
#
# Usage:
#   cd CIARD_Expansion_mobilenetv2_cifar10_v1
#   bash setup_models.sh
#
# This script will:
#   1. Download CIFAR-10 dataset
#   2. Download & convert the robust teacher (WRN-34-20, Rice2020 from RobustBench)
#   3. Download the natural teacher (ResNet-56 from chenyaofo/pytorch-cifar-models)

set -euo pipefail

echo "============================================"
echo "SARD Model Setup Script"
echo "Time: $(date)"
echo "============================================"

# Step 1: Download CIFAR-10
echo ""
echo "[Step 1] Downloading CIFAR-10 dataset..."
python3 -c "
import torchvision
torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
print('CIFAR-10 downloaded successfully.')
"

# Step 2: Download & convert robust teacher (WRN-34-20, Rice2020)
echo ""
echo "[Step 2] Downloading & converting robust teacher (WRN-34-20, Rice2020)..."
python3 -c "
import torch, sys, os
sys.path.insert(0, '.')
from robustbench import load_model

print('Loading RobustBench Rice2020Overfitting (WRN-34-20)...')
rb_model = load_model(model_name='Rice2020Overfitting', dataset='cifar10', threat_model='Linf')
rb_state = rb_model.state_dict()
print(f'RobustBench keys: {len(rb_state)}')

from cifar10_models.wideresnet import wideresnet
model = wideresnet(widen_factor=20, normalize=True)

# Build filtered state dict (skip mu/sigma, they're built-in buffers)
filtered = {}
for k, v in rb_state.items():
    if k in model.state_dict() and v.shape == model.state_dict()[k].shape:
        filtered[k] = v

model.load_state_dict(filtered, strict=False)
if missing := (set(model.state_dict().keys()) - set(filtered.keys())):
    non_subblock = [k for k in missing if 'sub_block1' not in k]
    if non_subblock:
        print(f'WARNING: Critical missing keys: {non_subblock[:5]}...')
        sys.exit(1)
    else:
        print(f'sub_block1 keys randomly initialized (expected, unused in forward)')
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/model_cifar_wrn.pt')
print(f'Saved robust teacher to models/model_cifar_wrn.pt ({os.path.getsize(\"models/model_cifar_wrn.pt\")/1024/1024:.1f} MB)')

# Quick accuracy check
import torchvision
from torchvision import transforms
transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)

model = model.cuda().eval()
correct = 0; total = 0
with torch.no_grad():
    for x, y in testloader:
        x, y = x.cuda(), y.cuda()
        out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
print(f'Robust teacher clean accuracy: {correct/total:.4f}')
"

# Step 3: Download natural teacher (ResNet-56)
echo ""
echo "[Step 3] Downloading natural teacher (ResNet-56 from chenyaofo)..."
mkdir -p models/nat_teacher_checkpoint
python3 -c "
import torch, sys, os
sys.path.insert(0, '.')

# Method 1: Try downloading from chenyaofo's repo
import urllib.request
url = 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/v1.0/cifar10-resnet56-951c35a1.pth'
dest = 'models/nat_teacher_checkpoint/cifar10_resnet56_chenyaofo.pt'
try:
    if not os.path.exists(dest):
        print(f'Downloading from {url}...')
        urllib.request.urlretrieve(url, dest)
    print(f'Downloaded: {dest} ({os.path.getsize(dest)/1024/1024:.1f} MB)')
except Exception as e:
    print(f'Download failed: {e}')
    print('Please manually download from: https://github.com/chenyaofo/pytorch-cifar-models')
    print('And save as: models/nat_teacher_checkpoint/cifar10_resnet56_chenyaofo.pt')
    sys.exit(1)

# Load and re-save in the code's expected format
from cifar10_nat_teacher_models import cifar10_resnet56
model = cifar10_resnet56(normalize=True)

def safe_load(path, map_location='cpu', weights_only=False):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)

sd = safe_load(dest, map_location='cpu', weights_only=False)

if isinstance(sd, dict) and 'state_dict' in sd:
    sd = sd['state_dict']
sd = {k.replace('module.', ''): v for k, v in sd.items()}
result = model.load_state_dict(sd, strict=False)
if result.missing_keys:
    print(f'Missing keys: {result.missing_keys[:5]}...')
if result.unexpected_keys:
    print(f'Unexpected keys: {result.unexpected_keys[:5]}...')
torch.save(model.state_dict(), 'models/nat_teacher_checkpoint/cifar10_resnnet56.pth')
print(f'Saved natural teacher to models/nat_teacher_checkpoint/cifar10_resnnet56.pth')

# Quick accuracy check
import torchvision
from torchvision import transforms
transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)

model = model.cuda().eval()
correct = 0; total = 0
with torch.no_grad():
    for x, y in testloader:
        x, y = x.cuda(), y.cuda()
        out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
nat_acc = correct / total
print(f'Natural teacher clean accuracy: {nat_acc:.4f}')
if nat_acc < 0.90:
    print('WARNING: Natural teacher accuracy below 90%! Check checkpoint loading.')
    sys.exit(1)
print('Natural teacher verified successfully.')
"

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "Teacher models ready:"
ls -la models/model_cifar_wrn.pt models/nat_teacher_checkpoint/cifar10_resnnet56.pth 2>/dev/null
echo ""
echo "To train SARD:"
echo "  python CIARD.py --sard_saa 1 --sard_rcd 1 --epochs 200 --prefix sard_200ep"
echo ""
echo "To train CIARD baseline:"
echo "  python CIARD.py --sard_saa 0 --sard_rcd 0 --epochs 200 --prefix baseline_200ep"
echo ""
echo "To evaluate:"
echo "  python fast_eval.py --checkpoint model/sard_200ep/student_best.pth --prefix sard_200ep"

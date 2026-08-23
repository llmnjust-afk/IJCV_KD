"""Convert RobustBench Rice2020 WRN-34-20 model to CIARD code format."""
import torch
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, '.')

from cifar10_models.wideresnet import wideresnet
from robustbench import load_model

print("Loading RobustBench Rice2020Overfitting (WRN-34-20)...")
rb_model = load_model(model_name='Rice2020Overfitting', dataset='cifar10', threat_model='Linf')
rb_state = rb_model.state_dict()

print(f"RB keys: {len(rb_state)}")

# Create code's WRN-34-20 with normalization
model = wideresnet(widen_factor=20, normalize=True)
code_state = model.state_dict()
print(f"Code keys: {len(code_state)}")

# Build filtered state dict
filtered = {}
skipped_rb = []
for k, v in rb_state.items():
    if k in code_state and v.shape == code_state[k].shape:
        filtered[k] = v
    else:
        skipped_rb.append(k)

missing = set(code_state.keys()) - set(filtered.keys())
print(f"Loaded: {len(filtered)}, Skipped RB keys: {skipped_rb}")
print(f"Missing code keys: {len(missing)}")
sub_block_missing = [k for k in missing if 'sub_block1' in k]
other_missing = [k for k in missing if 'sub_block1' not in k]
print(f"  sub_block1 (expected, unused): {len(sub_block_missing)}")
print(f"  Other missing: {len(other_missing)}")
if other_missing:
    print(f"  Other missing keys: {other_missing}")

# Load with strict=False (sub_block1 will be randomly initialized, which is fine)
model.load_state_dict(filtered, strict=False)
model = model.eval()

# Save
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/model_cifar_wrn.pt')
print(f"\nSaved to models/model_cifar_wrn.pt ({os.path.getsize('models/model_cifar_wrn.pt') / 1024 / 1024:.1f} MB)")

# Quick eval
import torchvision
from torchvision import transforms
transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)

model = model.cuda()
correct = 0
total = 0
with torch.no_grad():
    for x, y in testloader:
        x, y = x.cuda(), y.cuda()
        out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
print(f"Clean accuracy: {correct/total:.4f} ({correct}/{total})")

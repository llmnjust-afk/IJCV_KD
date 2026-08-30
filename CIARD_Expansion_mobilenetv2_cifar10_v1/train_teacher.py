"""Quick WRN-34-20 teacher training for CIARD experiments.
Trains with standard CE first, then switches to PGD adversarial training
to get a basic robust teacher checkpoint.

Note: For best results, use convert_rb_teacher.py to download the
pre-trained Rice2020 WRN-34-20 model from RobustBench instead.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CIARD_GPU", "0")

import sys
sys.path.insert(0, '.')
from cifar10_models.wideresnet import wideresnet

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

batch_size = 256
epochs_standard = 60
epochs_adv = 30
epsilon = 8 / 255.0
step_size = 2 / 255.0
attack_steps = 7

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)

model = wideresnet(widen_factor=20, normalize=True).cuda()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_standard + epochs_adv)
ce = nn.CrossEntropyLoss()

os.makedirs('models', exist_ok=True)

def attack_pgd(model, x, y, steps=7, step_size=2/255.0, epsilon=8/255.0):
    model.eval()
    x_adv = x.detach() + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)
    for _ in range(steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss = ce(model(x_adv), y)
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()
    return x_adv.detach()

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total

# Phase 1: standard training
print("=== Phase 1: Standard training ===")
for epoch in range(1, epochs_standard + 1):
    model.train()
    total_loss = 0
    for x, y in trainloader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out = model(x)
        loss = ce(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    if epoch % 10 == 0 or epoch == 1:
        acc = evaluate(model, testloader)
        print(f"Epoch {epoch}: loss={total_loss/len(trainloader):.4f}, clean_acc={acc:.4f}", flush=True)

# Save standard checkpoint
torch.save(model.state_dict(), 'models/model_cifar_wrn_standard.pt')
print(f"Saved standard teacher checkpoint (clean acc: {evaluate(model, testloader):.4f})", flush=True)

# Phase 2: adversarial training
print("\n=== Phase 2: Adversarial training ===")
for epoch in range(1, epochs_adv + 1):
    model.train()
    total_loss = 0
    for x, y in trainloader:
        x, y = x.cuda(), y.cuda()
        x_adv = attack_pgd(model, x, y, steps=attack_steps, step_size=step_size, epsilon=epsilon)
        optimizer.zero_grad()
        out = model(x_adv)
        loss = ce(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    if epoch % 5 == 0 or epoch == 1:
        clean_acc = evaluate(model, testloader)
        # Quick robust eval on 1000 samples
        model.eval()
        robust_correct = 0
        robust_total = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(testloader):
                if i >= 8:
                    break
                x, y = x.cuda(), y.cuda()
                x_adv = attack_pgd(model, x, y, steps=20, step_size=2/255.0, epsilon=epsilon)
                out = model(x_adv)
                robust_correct += (out.argmax(1) == y).sum().item()
                robust_total += y.size(0)
        model.train()
        print(f"Epoch {epoch}: loss={total_loss/len(trainloader):.4f}, clean={clean_acc:.4f}, robust(~)={robust_correct/robust_total:.4f}", flush=True)

# Save final adversarial checkpoint
torch.save(model.state_dict(), 'models/model_cifar_wrn.pt')
final_clean = evaluate(model, testloader)
print(f"\n=== Done! Final clean acc: {final_clean:.4f} ===", flush=True)
print(f"Saved to models/model_cifar_wrn.pt", flush=True)

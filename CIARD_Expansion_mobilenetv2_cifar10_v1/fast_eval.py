"""Fast evaluation script for SARD experiments.
Runs white-box + black-box attacks for quick feedback.
Usage: python fast_eval.py --checkpoint model/PREFIX/student_best.pth
"""
import os
import argparse
import json
import torch
import numpy as np
import torchvision
from torchvision import transforms
from loguru import logger
import torch.nn.functional as F

_parser = argparse.ArgumentParser(description="Fast evaluation for SARD experiments")
_parser.add_argument("--checkpoint", type=str, required=True, help="Path to student checkpoint .pth file")
_parser.add_argument("--prefix", type=str, default="", help="Experiment prefix (for logging)")
_args = _parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CIARD_GPU", "0")

import sys
sys.path.insert(0, '.')
from cifar10_models import *
from cifar10_nat_teacher_models import *

torch.manual_seed(0)

batch_size = 128
epsilon = 8 / 255.0

def safe_torch_load(path, map_location=torch.device('cpu'), weights_only=False):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)

transform_test = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

student = mobilenet_v2()
state_dict = safe_torch_load(_args.checkpoint, map_location=torch.device('cpu'), weights_only=False)["model"]
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
student.load_state_dict(new_state_dict)
student = student.cuda()
student.eval()

teacher1_path = 'models/model_cifar_wrn.pt'
teacher = wideresnet(widen_factor=20, normalize=True)
has_teacher = False
try:
    sd = safe_torch_load(teacher1_path, map_location=torch.device('cpu'), weights_only=False)
    new_sd = {k.replace('module.', ''): v for k, v in sd.items()}
    missing, unexpected = teacher.load_state_dict(new_sd, strict=False)
    if unexpected:
        logger.warning(f"Unexpected keys in teacher checkpoint: {unexpected[:5]}...")
    if missing:
        non_subblock = [k for k in missing if 'sub_block1' not in k]
        if non_subblock:
            logger.warning(f"Critical missing keys in teacher: {non_subblock[:5]}...")
    teacher = teacher.cuda()
    teacher.eval()
    has_teacher = True
except Exception as e:
    logger.warning(f"Could not load robust teacher: {e}")

logger.info(f"Eval: checkpoint={_args.checkpoint}, student=MobileNetV2, has_teacher={has_teacher}")

results = {}

def attack_pgd(model, data, labels, attack_iters=20, step_size=2/255.0, epsilon=8/255.0):
    ce = torch.nn.CrossEntropyLoss().cuda()
    x_adv = data.detach() + torch.zeros_like(data).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)
    for _ in range(attack_iters):
        x_adv.requires_grad_()
        model.zero_grad()
        logits = model(x_adv)
        loss = ce(logits, labels)
        loss.backward()
        grad = x_adv.grad.detach()
        x_adv = x_adv.detach() + step_size * torch.sign(grad)
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0, 1).detach()
    model.zero_grad()
    return x_adv

def attack_fgsm(model, data, labels, epsilon=8/255.0):
    ce = torch.nn.CrossEntropyLoss().cuda()
    data = data.detach().requires_grad_(True)
    model.zero_grad()
    logits = model(data)
    loss = ce(logits, labels)
    loss.backward()
    grad = data.grad.detach().sign()
    model.zero_grad()
    return torch.clamp(data + epsilon * grad, 0, 1).detach()

def attack_cw_inf(model, input, target, confidence=50, num_classes=10, epsilon=8/255, lr=2/255, steps=30):
    perturbation = torch.zeros_like(input).cuda().requires_grad_()
    for _ in range(steps):
        model.zero_grad()
        clamped_input = torch.clamp(input + perturbation, 0.0, 1.0)
        output = model(clamped_input)
        target_onehot = F.one_hot(target, num_classes=num_classes).float().cuda()
        real = torch.sum(target_onehot * output, dim=1)
        other = torch.max((1 - target_onehot) * output - target_onehot * 10000, dim=1)[0]
        loss = -torch.clamp(real - other + confidence, min=0.).mean()
        grad = torch.autograd.grad(loss, perturbation)[0]
        perturbation = (perturbation.detach() + lr * torch.sign(grad)).clamp(-epsilon, epsilon)
        perturbation = perturbation.requires_grad_()
    model.zero_grad()
    return torch.clamp(input + perturbation, 0, 1).detach()

def evaluate_attack(model, attack_fn, **kwargs):
    correct = 0
    total = 0
    for data, labels in testloader:
        data, labels = data.float().cuda(), labels.cuda()
        x_adv = attack_fn(model, data, labels, **kwargs)
        with torch.no_grad():
            logits = model(x_adv)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total

def evaluate_clean(model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.float().cuda(), labels.cuda()
            logits = model(data)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total

clean_acc = evaluate_clean(student)
logger.info(f"Clean Accuracy: {clean_acc:.4f}")
results['clean'] = clean_acc

pgd_trades = evaluate_attack(student, attack_pgd, attack_iters=20, step_size=0.003, epsilon=epsilon)
logger.info(f"WB PGD-20 (step=0.003): {pgd_trades:.4f}")
results['wb_pgd_trades'] = pgd_trades

pgd_sat = evaluate_attack(student, attack_pgd, attack_iters=20, step_size=2/255.0, epsilon=epsilon)
logger.info(f"WB PGD-20 (step=2/255): {pgd_sat:.4f}")
results['wb_pgd_sat'] = pgd_sat

fgsm = evaluate_attack(student, attack_fgsm, epsilon=epsilon)
logger.info(f"WB FGSM: {fgsm:.4f}")
results['wb_fgsm'] = fgsm

cw = evaluate_attack(student, attack_cw_inf, confidence=50, num_classes=10, epsilon=epsilon, lr=2/255, steps=30)
logger.info(f"WB CW L_inf: {cw:.4f}")
results['wb_cw'] = cw

if has_teacher:
    bb_pgd_correct = 0
    bb_pgd_total = 0
    for data, labels in testloader:
        data, labels = data.float().cuda(), labels.cuda()
        x_adv = attack_pgd(teacher, data, labels, attack_iters=20, step_size=0.003, epsilon=epsilon)
        with torch.no_grad():
            logits = student(x_adv)
        bb_pgd_correct += (logits.argmax(1) == labels).sum().item()
        bb_pgd_total += labels.size(0)
    bb_pgd = bb_pgd_correct / bb_pgd_total
    logger.info(f"BB PGD-20: {bb_pgd:.4f}")
    results['bb_pgd'] = bb_pgd

    bb_cw_correct = 0
    bb_cw_total = 0
    for data, labels in testloader:
        data, labels = data.float().cuda(), labels.cuda()
        x_adv = attack_cw_inf(teacher, data, labels, confidence=50, num_classes=10, epsilon=epsilon, lr=2/255, steps=30)
        with torch.no_grad():
            logits = student(x_adv)
        bb_cw_correct += (logits.argmax(1) == labels).sum().item()
        bb_cw_total += labels.size(0)
    bb_cw = bb_cw_correct / bb_cw_total
    logger.info(f"BB CW L_inf: {bb_cw:.4f}")
    results['bb_cw'] = bb_cw
else:
    logger.warning("Black-box attacks SKIPPED: teacher not loaded")

logger.info("="*60)
summary_parts = [f"clean={clean_acc:.4f}", f"pgd_trades={pgd_trades:.4f}", f"pgd_sat={pgd_sat:.4f}", f"fgsm={fgsm:.4f}", f"cw={cw:.4f}"]
if has_teacher:
    summary_parts.append(f"bb_pgd={results['bb_pgd']:.4f}")
    summary_parts.append(f"bb_cw={results['bb_cw']:.4f}")
logger.info("SUMMARY: " + " ".join(summary_parts))
logger.info("EVAL_COMPLETE")

results['has_teacher'] = has_teacher
results['checkpoint'] = _args.checkpoint
out_path = f"eval_{_args.prefix or 'results'}.json" if _args.prefix else "eval_results.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
logger.info(f"Results saved to {out_path}")

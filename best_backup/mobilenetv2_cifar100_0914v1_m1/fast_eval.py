"""Fast evaluation script for SARD experiments.
Runs white-box + black-box attacks for quick feedback.
Usage: python fast_eval.py --checkpoint model/PREFIX/student_best.pth
"""
import os
import argparse
import datetime
import hashlib
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

def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if not torch.cuda.is_available():
    raise RuntimeError("Fast evaluation requires a CUDA GPU")
if not os.path.isfile(_args.checkpoint):
    raise FileNotFoundError(_args.checkpoint)

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
if not os.path.isfile(teacher1_path):
    raise FileNotFoundError(teacher1_path)
teacher = wideresnet()
sd = safe_torch_load(teacher1_path, map_location=torch.device('cpu'), weights_only=False)
new_sd = {k.replace('module.', ''): v for k, v in sd.items()}
teacher.load_state_dict(new_sd, strict=True)
teacher = teacher.cuda()
teacher.eval()
has_teacher = True

logger.info("""Fast evaluation config:
checkpoint: {}
checkpoint_sha256: {}
student: MobileNetV2
blackbox_teacher_arch: WRN-34-10 raw
blackbox_teacher_checkpoint: {}
blackbox_teacher_sha256: {}
test_samples: {}
batch_size: {}
seed: 0
""".format(
    os.path.realpath(_args.checkpoint), checkpoint_sha256(_args.checkpoint),
    os.path.realpath(teacher1_path), checkpoint_sha256(teacher1_path),
    len(testset), batch_size))

results = {}

def attack_pgd(model, data, labels, attack_iters=20, step_size=2/255.0, epsilon=8/255.0):
    x_adv = data.detach() + torch.zeros_like(data).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)
    for _ in range(attack_iters):
        x_adv.requires_grad_()
        model.zero_grad()
        logits = model(x_adv)
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, x_adv)[0].detach()
        x_adv = x_adv.detach() + step_size * torch.sign(grad)
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0, 1).detach()
    model.zero_grad()
    return x_adv

def attack_fgsm(model, data, labels, epsilon=8/255.0):
    data = data.detach().requires_grad_(True)
    model.zero_grad()
    logits = model(data)
    loss = F.cross_entropy(logits, labels)
    grad = torch.autograd.grad(loss, data)[0].detach().sign()
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
        projected = torch.clamp(input + perturbation, 0.0, 1.0)
        perturbation = (projected - input).detach().requires_grad_()
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

logger.info("="*60)
summary_parts = [f"clean={clean_acc:.4f}", f"pgd_trades={pgd_trades:.4f}", f"pgd_sat={pgd_sat:.4f}", f"fgsm={fgsm:.4f}", f"cw={cw:.4f}"]
summary_parts.append(f"bb_pgd={results['bb_pgd']:.4f}")
summary_parts.append(f"bb_cw={results['bb_cw']:.4f}")
logger.info("SUMMARY: " + " ".join(summary_parts))

results['has_teacher'] = has_teacher
results['checkpoint'] = _args.checkpoint
required_metrics = {'clean', 'wb_pgd_trades', 'wb_pgd_sat', 'wb_fgsm', 'wb_cw', 'bb_pgd', 'bb_cw'}
missing_metrics = sorted(required_metrics.difference(results))
if missing_metrics:
    raise RuntimeError("Fast evaluation incomplete: {}".format(missing_metrics))
result_tag = os.environ.get(
    "SLURM_JOB_ID", datetime.datetime.now().strftime("manual_%Y%m%d_%H%M%S"))
out_path = os.path.join(
    os.path.dirname(_args.checkpoint),
    "fast_eval_{}_{}.json".format(_args.prefix or "results", result_tag))
with open(out_path, 'x') as f:
    json.dump(results, f, indent=2)
logger.info(f"Results saved to {out_path}")
logger.info("EVAL_COMPLETE")

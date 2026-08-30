import datetime
import hashlib
import json
import os

import torch
import torch.nn.functional as F
import torchattacks
import torchvision
from autoattack import AutoAttack
from loguru import logger
from torchvision import transforms

from cifar10_models import mobilenet_v2, wideresnet


VARIANT_NAME = "mobilenetv2_cifar10_0830_source"
EVAL_TARGET = "student_best"
CHECKPOINT = "model/Cifar10_MobileNetV2_tm010_repeat0620/student_best.pth"
ROBUST_TEACHER_CHECKPOINT = "models/model_cifar_wrn.pt"
RESULTS_REQUIRED = (
    "clean", "autoattack", "wb_pgd_trades", "wb_pgd_sat", "wb_fgsm",
    "wb_cw", "bb_pgd_trades", "bb_square", "bb_cw",
)


def safe_torch_load(path, map_location=torch.device("cpu"), weights_only=False):
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


def strip_module_prefix(state_dict):
    return {key.replace("module.", ""): value for key, value in state_dict.items()}


def attack_pgd(model, data, labels, attack_iters=20, step_size=2/255.0,
               epsilon=8/255.0):
    x_adv = data.detach() + torch.empty_like(data).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for _ in range(attack_iters):
        x_adv.requires_grad_(True)
        model.zero_grad()
        loss = F.cross_entropy(model(x_adv), labels)
        gradient = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + step_size * gradient.detach().sign()
        x_adv = torch.max(torch.min(x_adv, data + epsilon), data - epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    model.zero_grad()
    return x_adv


def attack_fgsm(model, data, labels, epsilon=8/255.0):
    x_adv = data.detach().requires_grad_(True)
    model.zero_grad()
    loss = F.cross_entropy(model(x_adv), labels)
    gradient = torch.autograd.grad(loss, x_adv)[0]
    model.zero_grad()
    return torch.clamp(data + epsilon * gradient.detach().sign(), 0.0, 1.0).detach()


def attack_cw_inf(model, data, labels, confidence=50, num_classes=10,
                  epsilon=8/255.0, step_size=2/255.0, steps=30):
    perturbation = torch.zeros_like(data).requires_grad_(True)
    target_onehot = F.one_hot(labels, num_classes=num_classes).float()
    for _ in range(steps):
        model.zero_grad()
        x_adv = torch.clamp(data + perturbation, 0.0, 1.0)
        output = model(x_adv)
        real = torch.sum(target_onehot * output, dim=1)
        other = torch.max((1.0 - target_onehot) * output - target_onehot * 10000.0, dim=1)[0]
        objective = -torch.clamp(real - other + confidence, min=0.0).mean()
        gradient = torch.autograd.grad(objective, perturbation)[0]
        perturbation = perturbation.detach() + step_size * gradient.detach().sign()
        perturbation = torch.clamp(perturbation, -epsilon, epsilon)
        projected = torch.clamp(data + perturbation, 0.0, 1.0)
        perturbation = (projected - data).detach().requires_grad_(True)
    model.zero_grad()
    return torch.clamp(data + perturbation.detach(), 0.0, 1.0)


def evaluate_clean(model, dataloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            correct += (model(data).argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def evaluate_attack(source_model, target_model, dataloader, device, attack_fn, **kwargs):
    correct = 0
    total = 0
    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)
        adversarial_data = attack_fn(source_model, data, labels, **kwargs)
        with torch.no_grad():
            correct += (target_model(adversarial_data).argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


def evaluate_autoattack(model, dataloader, device, epsilon=8/255.0):
    inputs, labels = [], []
    for batch_inputs, batch_labels in dataloader:
        inputs.append(batch_inputs)
        labels.append(batch_labels)
    x_test = torch.cat(inputs, dim=0).to(device)
    y_test = torch.cat(labels, dim=0).to(device)
    adversary = AutoAttack(model, norm="Linf", eps=epsilon, version="standard", verbose=True)
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=128)
    with torch.no_grad():
        return (model(x_adv).argmax(1) == y_test).float().mean().item()


def evaluate_square(model, dataloader, device, epsilon=8/255.0, queries=100):
    attack = torchattacks.Square(model, norm="Linf", eps=epsilon, n_queries=queries)
    correct = 0
    total = 0
    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)
        adversarial_data = attack(data, labels)
        with torch.no_grad():
            correct += (model(adversarial_data).argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Full evaluation requires a CUDA GPU")
    for required_path in (CHECKPOINT, ROBUST_TEACHER_CHECKPOINT):
        if not os.path.isfile(required_path):
            raise FileNotFoundError(required_path)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda:0")

    student = mobilenet_v2()
    checkpoint_payload = safe_torch_load(CHECKPOINT, weights_only=False)
    if "model" not in checkpoint_payload:
        raise KeyError("Student checkpoint does not contain a 'model' state_dict")
    student.load_state_dict(strip_module_prefix(checkpoint_payload["model"]), strict=True)
    student = student.to(device).eval()

    robust_teacher = wideresnet()
    teacher_state = strip_module_prefix(
        safe_torch_load(ROBUST_TEACHER_CHECKPOINT, weights_only=False))
    robust_teacher.load_state_dict(teacher_state, strict=True)
    robust_teacher = robust_teacher.to(device).eval()

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor()]))
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=0)

    student_hash = checkpoint_sha256(CHECKPOINT)
    teacher_hash = checkpoint_sha256(ROBUST_TEACHER_CHECKPOINT)
    logger.info("""CIARD resolved evaluation config:
variant: {}
eval_target: {}
checkpoint: {}
checkpoint_sha256: {}
dataset: CIFAR10
test_samples: {}
student: MobileNetV2
num_classes: 10
batch_size: 128
seed: 0
blackbox_teacher_arch: WRN-34-10 raw
blackbox_teacher_checkpoint: {}
blackbox_teacher_sha256: {}
autoattack: standard Linf epsilon=8/255
whitebox_pgd_trades: steps=20 step_size=0.003 epsilon=8/255
whitebox_pgd_sat: steps=20 step_size=2/255 epsilon=8/255
whitebox_fgsm: epsilon=8/255
whitebox_cw: steps=30 step_size=2/255 epsilon=8/255 confidence=50
blackbox_pgd_trades: source=WRN-34-10 steps=20 step_size=0.003 epsilon=8/255
blackbox_square: queries=100 epsilon=8/255
blackbox_cw: source=WRN-34-10 steps=30 step_size=2/255 epsilon=8/255 confidence=50
""".format(
        VARIANT_NAME, EVAL_TARGET, os.path.realpath(CHECKPOINT), student_hash,
        len(testset), os.path.realpath(ROBUST_TEACHER_CHECKPOINT), teacher_hash))

    results = {
        "variant": VARIANT_NAME,
        "eval_target": EVAL_TARGET,
        "checkpoint": os.path.realpath(CHECKPOINT),
        "checkpoint_sha256": student_hash,
        "blackbox_teacher_checkpoint": os.path.realpath(ROBUST_TEACHER_CHECKPOINT),
        "blackbox_teacher_sha256": teacher_hash,
    }
    results["clean"] = evaluate_clean(student, testloader, device)
    logger.info("student clean acc: {:.4f}", results["clean"])
    results["autoattack"] = evaluate_autoattack(student, testloader, device)
    logger.info("student robust acc under AutoAttack: {:.4f}", results["autoattack"])
    results["wb_pgd_trades"] = evaluate_attack(
        student, student, testloader, device, attack_pgd,
        attack_iters=20, step_size=0.003, epsilon=8/255.0)
    logger.info("student robust acc under white-box PGD_trades Attack: {:.4f}", results["wb_pgd_trades"])
    results["wb_pgd_sat"] = evaluate_attack(
        student, student, testloader, device, attack_pgd,
        attack_iters=20, step_size=2/255.0, epsilon=8/255.0)
    logger.info("student robust acc under white-box PGD_sat Attack: {:.4f}", results["wb_pgd_sat"])
    results["wb_fgsm"] = evaluate_attack(
        student, student, testloader, device, attack_fgsm, epsilon=8/255.0)
    logger.info("student robust acc under white-box FGSM Attack: {:.4f}", results["wb_fgsm"])
    results["wb_cw"] = evaluate_attack(
        student, student, testloader, device, attack_cw_inf,
        confidence=50, num_classes=10, epsilon=8/255.0, step_size=2/255.0, steps=30)
    logger.info("student robust acc under white-box CW L_inf: {:.4f}", results["wb_cw"])
    results["bb_pgd_trades"] = evaluate_attack(
        robust_teacher, student, testloader, device, attack_pgd,
        attack_iters=20, step_size=0.003, epsilon=8/255.0)
    logger.info("student robust acc under black-box PGD_trades Attack: {:.4f}", results["bb_pgd_trades"])
    results["bb_square"] = evaluate_square(student, testloader, device)
    logger.info("student robust acc under black-box Square Attack: {:.4f}", results["bb_square"])
    results["bb_cw"] = evaluate_attack(
        robust_teacher, student, testloader, device, attack_cw_inf,
        confidence=50, num_classes=10, epsilon=8/255.0, step_size=2/255.0, steps=30)
    logger.info("student robust acc under black-box CW L_inf: {:.4f}", results["bb_cw"])

    missing = [key for key in RESULTS_REQUIRED if key not in results]
    if missing:
        raise RuntimeError("Evaluation incomplete; missing metrics: {}".format(missing))

    result_tag = os.environ.get(
        "SLURM_JOB_ID", datetime.datetime.now().strftime("manual_%Y%m%d_%H%M%S"))
    result_path = os.path.join(
        os.path.dirname(CHECKPOINT), "eval_{}_{}.json".format(EVAL_TARGET, result_tag))
    with open(result_path, "x") as result_file:
        json.dump(results, result_file, indent=2, sort_keys=True)
    logger.info("structured_results: {}", os.path.realpath(result_path))
    logger.info("EVAL_COMPLETE")


if __name__ == "__main__":
    main()

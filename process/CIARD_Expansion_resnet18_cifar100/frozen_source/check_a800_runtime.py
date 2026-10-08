"""CUDA smoke check, invoked only inside the user's A800 training/evaluation job.

Runs in a separate process before CIARD/attack_eval, without changing their RNG,
model state, losses or attack protocol. Does not read test data or save weights.
"""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected-gpus', type=int, choices=(1,), required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run this CUDA check only inside a manually submitted Slurm job')
    import torch
    import torch.nn.functional as F
    from cifar10_models.resnet import ResNet, BasicBlock
    if not torch.cuda.is_available() or torch.cuda.device_count() != args.expected_gpus:
        raise RuntimeError('Unexpected Slurm-visible GPU allocation')
    architectures = torch.cuda.get_arch_list()
    if 'sm_80' not in architectures:
        raise RuntimeError('This PyTorch build lacks native sm_80 support for A800')
    print('A800_RUNTIME torch={} cuda={} cudnn={} architectures={}'.format(
        torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), architectures))
    driver = Path('/proc/driver/nvidia/version')
    if driver.is_file():
        print('A800_DRIVER ' + driver.read_text().replace('\n', ' '))
    for device in range(args.expected_gpus):
        info = torch.cuda.get_device_properties(device)
        if 'A800' not in info.name or (info.major, info.minor) != (8, 0):
            raise RuntimeError('Expected A800 with compute capability 8.0')
        print('A800_DEVICE index={} name={} memory_bytes={}'.format(device, info.name, info.total_memory))
    torch.backends.cudnn.deterministic = True
    model = torch.nn.DataParallel(ResNet(BasicBlock, [2, 2, 2, 2], num_classes=100).cuda())
    images = torch.rand(128, 3, 32, 32, device='cuda', requires_grad=True)
    labels = torch.arange(128, device='cuda') % 100
    model.eval()
    gradient, = torch.autograd.grad(F.cross_entropy(model(images), labels), images)
    if not torch.isfinite(gradient).all():
        raise RuntimeError('Nonfinite adversarial input gradient')
    adversarial = (images.detach() + (2 / 255.0) * gradient.sign()).clamp(0, 1)
    model.train()
    loss = F.cross_entropy(model(adversarial), labels)
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError('Invalid model forward/backward result')
    for device in range(args.expected_gpus):
        torch.cuda.synchronize(device)
    print('A800_RUNTIME_OK gpus={} batch=128 input_gradient=passed backward=passed'.format(args.expected_gpus))


if __name__ == '__main__':
    main()

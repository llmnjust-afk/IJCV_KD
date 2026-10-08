"""CPU preflight by default; CUDA is checked only by the user-submitted scripts."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_hashes():
    return {str(path): sha256(path) for path in sorted(Path('.').rglob('*.py'))
            if not any(part in ('data', 'models', 'model', 'logs', '__pycache__') for part in path.parts)}


def literals(path):
    out = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        out[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
    return out


def identity():
    train, evaluation = literals('CIARD.py'), literals('attack_eval.py')
    checkpoint = Path('model') / train['prefix'] / 'student_best.pth'
    if checkpoint.resolve() != Path(evaluation['path']).resolve():
        raise ValueError('Training prefix and fixed evaluator checkpoint disagree')
    if train['VARIANT_NAME'] != evaluation['variant_name']:
        raise ValueError('Training and evaluation variants disagree')
    return train['VARIANT_NAME'], checkpoint


def require_slurm_success(job):
    if not re.fullmatch(r'[0-9]+', str(job)):
        raise ValueError('Invalid Slurm job ID')
    result = subprocess.run(['sacct', '-j', str(job), '--format=JobIDRaw,State,ExitCode', '-P', '-n'],
                            check=True, text=True, stdout=subprocess.PIPE)
    rows = [row.split('|') for row in result.stdout.splitlines()]
    if not any(row[:3] == [str(job), 'COMPLETED', '0:0'] for row in rows):
        raise ValueError('Slurm has not confirmed COMPLETED / 0:0 for job ' + str(job))


def checkpoint_epoch(checkpoint, variant):
    import torch
    from cifar10_models.resnet import ResNet, BasicBlock
    from cifar10_models.mobilenet_v2 import MobileNetV2
    factory = (lambda: ResNet(BasicBlock, [2, 2, 2, 2], num_classes=100)) if variant.startswith('resnet18_') else (lambda: MobileNetV2(num_classes=100))
    state = torch.load(checkpoint, map_location='cpu')
    model = factory()
    model.load_state_dict({key.replace('module.', ''): value for key, value in state['model'].items()}, strict=True)
    epoch = int(state['epoch'])
    if not 1 <= epoch <= 300:
        raise ValueError('Unexpected selected checkpoint epoch')
    return epoch


def preflight(use_cuda=False, expected_gpus=2):
    import torch
    import torchvision
    import loguru, numpy, torchattacks, autoattack
    from cifar100_teacher import robust_teacher, natural_teacher, CIFAR100_MEAN, CIFAR100_STD
    from robustbench.model_zoo.architectures import dm_wide_resnet
    if sha256(dm_wide_resnet.__file__) != '244e53a8a94e1166948a94e2ac2769af8ba29ad065b9afcbb0de15ed51d3cc7c':
        raise ValueError("RobustBench teacher architecture SHA256 mismatch")
    from awp_consistency import validate_config, method_enabled
    settings = literals('CIARD.py')
    cfg = settings['CFG']
    validate_config(cfg)
    if method_enabled(cfg, cfg['method_start'] + 1) and not settings['USE_CIARDPP']:
        raise ValueError('0909 methods require USE_CIARDPP=True')
    if torch.__version__ != '1.10.0+cu113' or torchvision.__version__ != '0.11.1+cu113':
        raise RuntimeError('Expected ciard environment: torch 1.10.0+cu113 / torchvision 0.11.1+cu113')
    print('preflight_torch={} torchvision={} cuda_build={}'.format(
        torch.__version__, torchvision.__version__, torch.version.cuda))
    specifications = [
        ('models/cifar100_linf_wrn70-16_without.pt', '3114df6b9d5adf9f275e8fea5a91b71c61df78a568ad31544b6950807d595c8c', robust_teacher, 'logits.weight', (100, 1024), 'WRN-70-16'),
        ('models/nat_teacher_checkpoint/cifar100_wrn_22_6.pth', 'c91c5bf8b5f6c74c427d9a88815c00f98b73a4d1109fb4508b985ae86919e935', natural_teacher, 'logits.weight', (100, 384), 'WRN-22-6')]
    for path, digest, factory, key, shape, architecture in specifications:
        if sha256(path) != digest:
            raise ValueError('Teacher SHA256 mismatch: ' + path)
        state = torch.load(path, map_location='cpu')
        state = {key.replace('module.', ''): value for key, value in state.items()}
        if tuple(state[key].shape) != shape or any(k.endswith(('mu', 'sigma')) for k in state):
            raise ValueError('Unexpected teacher architecture / normalization')
        model = factory()
        model.load_state_dict(state, strict=True)
        if not torch.equal(model.mean.flatten(), torch.tensor(CIFAR100_MEAN)) or not torch.equal(model.std.flatten(), torch.tensor(CIFAR100_STD)):
            raise ValueError("Unexpected CIFAR100 teacher normalization")
        print('teacher={} checkpoint={} bytes={} sha256={} shape={} input=raw model_internal_normalization=CIFAR100'.format(
            architecture, Path(path).resolve(), Path(path).stat().st_size, digest, shape))
        del model, state
    train = torchvision.datasets.CIFAR100(root='./data', train=True, download=False)
    test = torchvision.datasets.CIFAR100(root='./data', train=False, download=False)
    if (len(train), len(test)) != (50000, 10000):
        raise ValueError('Unexpected CIFAR-100 split sizes')
    if len(train.classes) != 100 or len(test.classes) != 100 or set(train.targets) != set(range(100)) or set(test.targets) != set(range(100)):
        raise ValueError('Expected 100 classes and labels 0..99')
    print('preflight_dataset=CIFAR100 classes=100 samples=train:50000 test:10000')
    identity()
    if use_cuda:
        if not torch.cuda.is_available() or torch.cuda.device_count() != expected_gpus:
            raise RuntimeError('Expected exactly {} Slurm-visible CUDA GPUs'.format(expected_gpus))
        for device in range(expected_gpus):
            if '4090' not in torch.cuda.get_device_name(device):
                raise RuntimeError('Expected requested 4090 GPU')
            probe = torch.empty(1, device='cuda:' + str(device))
            print('preflight_gpu={} name={}'.format(device, torch.cuda.get_device_name(device)))
            del probe
    elif torch.cuda.is_initialized():
        raise RuntimeError('CPU preparation unexpectedly initialized CUDA')
    print('PRECHECK_OK mode=' + ('cuda' if use_cuda else 'cpu'))


def start_training():
    _, checkpoint = identity()
    if checkpoint.parent.exists():
        raise ValueError('Refusing existing training prefix: ' + str(checkpoint.parent))
    print('SOURCE_FILES_JSON=' + json.dumps(source_hashes(), sort_keys=True))


def finish_training(log, job):
    variant, checkpoint = identity()
    text = Path(log).read_text()
    rows = [line.split('=', 1)[1] for line in text.splitlines() if line.startswith('SOURCE_FILES_JSON=')]
    if len(rows) != 1 or json.loads(rows[0]) != source_hashes():
        raise ValueError('Source files changed during training')
    if 'PRECHECK_OK mode=cuda' not in text or 'Traceback (most recent call last)' in text:
        raise ValueError('Training log lacks CUDA preflight or contains an exception')
    if not checkpoint.is_file() or not checkpoint.stat().st_size:
        raise ValueError('Missing completed best checkpoint')
    completion = {'variant': variant, 'job_id': str(job), 'checkpoint': str(checkpoint.resolve()),
        'checkpoint_sha256': sha256(checkpoint), 'epoch': checkpoint_epoch(checkpoint, variant),
        'source_hashes': source_hashes(),
        'training_log': str(Path(log).resolve())}
    destination = checkpoint.parent / 'training_complete.json'
    with destination.open('x') as stream:
        json.dump(completion, stream, indent=2)
    print('TRAIN_COMPLETE variant={} checkpoint={} checkpoint_sha256={}'.format(
        variant, checkpoint.resolve(), completion['checkpoint_sha256']))


def verify_training(check_slurm=True):
    variant, checkpoint = identity()
    completion = json.loads((checkpoint.parent / 'training_complete.json').read_text())
    if (completion['variant'] != variant or completion['checkpoint'] != str(checkpoint.resolve())
            or completion['checkpoint_sha256'] != sha256(checkpoint)
            or completion['source_hashes'] != source_hashes()):
        raise ValueError('Completed training identity, checkpoint or sources no longer match')
    marker = 'TRAIN_COMPLETE variant={} checkpoint={} checkpoint_sha256={}'.format(
        variant, checkpoint.resolve(), completion['checkpoint_sha256'])
    if marker not in Path(completion['training_log']).read_text().splitlines():
        raise ValueError('Missing final training completion marker')
    if check_slurm:
        require_slurm_success(completion['job_id'])
    if checkpoint_epoch(checkpoint, variant) != completion['epoch']:
        raise ValueError('Selected checkpoint epoch changed')
    print('TRAIN_COMPLETION_VERIFIED job=' + completion['job_id'])
    return completion


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['preflight', 'start', 'finish', 'verify'])
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--expected-gpus', type=int, choices=[1, 2], default=2)
    parser.add_argument('--log')
    parser.add_argument('--job')
    args = parser.parse_args()
    if args.action == 'preflight':
        preflight(args.cuda, args.expected_gpus)
    elif args.action == 'start':
        start_training()
    elif args.action == 'finish':
        finish_training(args.log, args.job)
    else:
        verify_training()

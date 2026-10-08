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


GROUP = Path(__file__).resolve().parent
PROTOCOL = 'paired_seed0_historical_attacks'
SELECTION = 'fixed_epoch190_no_test_selection'
STATE_KEYS = {'student', 'ema_student', 'teacher', 'optimizer', 'teacher_optimizer', 'dynamics', 'rng'}
METHODS = {
    'H1': ('resnet18_cifar100_nb_l1p5_w40_s0', 1.5, 40),
    'H2': ('resnet18_cifar100_nb_l2_w40_s0', 2., 40),
    'H3': ('resnet18_cifar100_nb_l3_w40_s0', 3., 40),
    'H4': ('resnet18_cifar100_nb_l2_w20_s0', 2., 20)}


def source_hashes():
    return {str(path.relative_to(GROUP)): sha256(path) for path in sorted(GROUP.rglob('*.py'))
            if not any(part in ('data', 'models', 'model', 'logs', '__pycache__')
                       for part in path.relative_to(GROUP).parts)}


def script_hashes():
    return {path.name: sha256(path) for path in sorted(GROUP.glob('*.sbatch'))}


def literals(path):
    out = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        class ResolveConstants(ast.NodeTransformer):
                            def visit_Name(self, item):
                                if item.id in out and isinstance(out[item.id], (str, int, float, bool)):
                                    return ast.copy_location(ast.Constant(value=out[item.id]), item)
                                return item
                        out[target.id] = ast.literal_eval(ResolveConstants().visit(node.value))
                    except (ValueError, TypeError):
                        pass
    return out


def settings():
    manifest = json.loads((GROUP.parent / 'experiment_manifest.json').read_text())
    entries = [item for item in manifest['experiments'] if item['name'] == GROUP.name]
    if len(entries) != 1:
        raise ValueError('Group is not uniquely identified in manifest')
    entry = entries[0]
    if (manifest['batch'] != '1004-cifar100-binary-v1'
            or manifest['protocol'] != PROTOCOL or manifest['selection_protocol'] != SELECTION
            or type(manifest['epochs']) is not int or manifest['epochs'] != 190):
        raise ValueError('Wrong frozen batch protocol or endpoint')
    if (entry['id'] not in METHODS or type(entry['training_seed']) is not int
            or entry['training_seed'] != 0 or entry['error_swap'] is not False):
        raise ValueError('Wrong experiment seed or disabled error-swap identity')
    name, binary, warmup = METHODS[entry['id']]
    method = 'natural_binary_outer'
    if (entry['method'] != method or entry['name'] != name
            or entry['prefix'] != 'Cifar100_ResNet18_1004_binary_v1_' + entry['id']):
        raise ValueError('Wrong fixed group, method or prefix')
    train = literals(GROUP / 'CIARD.py')
    expected_cfg = {'robust_kd_error_swap': False, 'clean_ce_transfer_ratio': 0.,
                    'clean_ce_preserve_loss': False, 'natural_binary_weight': binary,
                    'natural_binary_awp': False, 'natural_binary_start': 120,
                    'natural_binary_warmup': warmup}
    for key, value in expected_cfg.items():
        actual = train['CFG'][key]
        numeric = type(value) is float and type(actual) in (int, float)
        if (not numeric and type(actual) is not type(value)) or actual != value:
            raise ValueError('Wrong fixed method configuration: ' + key)
    return manifest, entry, train


def identity():
    _, entry, train = settings()
    evaluation = literals(GROUP / 'attack_eval.py')
    checkpoint = GROUP / 'model' / entry['prefix'] / 'student_epoch190.pth'
    if (checkpoint.resolve() != Path(entry['checkpoint']).resolve()
            or checkpoint.resolve() != Path(evaluation['path']).resolve()
            or train['prefix'] != entry['prefix'] or train['VARIANT_NAME'] != entry['name']
            or evaluation['variant_name'] != entry['name']
            or type(train['TRAINING_SEED']) is not int or train['TRAINING_SEED'] != entry['training_seed']
            or train['ERROR_SWAP'] is not False or train['CLEAN_METHOD'] != entry['method']):
        raise ValueError('Training, manifest and evaluator identity disagree')
    return entry['name'], checkpoint


def verify_frozen():
    _, entry, _ = settings()
    manifest_path = GROUP.parent / 'experiment_manifest.json'
    digest = sha256(manifest_path)
    if (GROUP.parent / 'preparation/experiment_manifest.sha256').read_text().split()[0] != digest:
        raise ValueError('Frozen manifest changed')
    current, scripts = source_hashes(), script_hashes()
    if current != entry['prepared_python_sha256'] or scripts != entry['prepared_script_sha256']:
        raise ValueError('Prepared sources or scripts changed')
    identity()
    return {'source_hashes': current, 'script_hashes': scripts, 'manifest_sha256': digest}


def require_slurm_success(job):
    if not re.fullmatch(r'[0-9]+', str(job)):
        raise ValueError('Invalid Slurm job ID')
    result = subprocess.run(['sacct', '-j', str(job), '--format=JobIDRaw,State,ExitCode', '-P', '-n'],
                            check=True, text=True, stdout=subprocess.PIPE)
    rows = [row.split('|') for row in result.stdout.splitlines()]
    if not any(row[:3] == [str(job), 'COMPLETED', '0:0'] for row in rows):
        raise ValueError('Slurm has not confirmed COMPLETED / 0:0 for job ' + str(job))


def training_state_metadata():
    from training_state import state_sha256
    _, entry, _ = settings()
    _, checkpoint = identity()
    hashes, fingerprints, states = {}, {}, {}
    for epoch in (120, 158, 190):
        path = checkpoint.parent / ('training_state_{}.json'.format(epoch))
        state = json.loads(path.read_text())
        if (type(state['epoch']) is not int or state['epoch'] != epoch
                or state['variant'] != entry['name'] or type(state['training_seed']) is not int
                or state['training_seed'] != entry['training_seed']
                or type(state['error_swap']) is not bool or state['error_swap'] != entry['error_swap']
                or state['clean_method'] != entry['method']
                or set(state['state_sha256']) != STATE_KEYS
                or any(not re.fullmatch(r'[0-9a-f]{64}', value) for value in state['state_sha256'].values())
                or not re.fullmatch(r'[0-9a-f]{64}', state['fingerprint_sha256'])):
            raise ValueError('Invalid training state sidecar identity or hashes')
        if state_sha256({'epoch': epoch, 'state_sha256': state['state_sha256']}) != state['fingerprint_sha256']:
            raise ValueError('Training fingerprint does not match component hashes')
        hashes[str(epoch)], fingerprints[str(epoch)], states[epoch] = sha256(path), state['fingerprint_sha256'], state
    return {'training_state_sha256': hashes, 'training_fingerprint_sha256': fingerprints}, states


def checkpoint_epoch(checkpoint, variant):
    import torch
    from cifar10_models.resnet import ResNet, BasicBlock
    from training_state import state_sha256
    _, entry, _ = settings()
    state = torch.load(checkpoint, map_location='cpu')
    if (type(state['epoch']) is not int or state['epoch'] != 190 or state['variant'] != variant
            or type(state['training_seed']) is not int or state['training_seed'] != entry['training_seed']
            or type(state['error_swap']) is not bool or state['error_swap'] != entry['error_swap']
            or state['clean_method'] != entry['method']
            or state['selection_protocol'] != SELECTION or state['checkpoint_role'] != 'ema_student'):
        raise ValueError('Expected fixed epoch190 EMA checkpoint with exact experiment metadata')
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=100)
    model.load_state_dict({key.replace('module.', ''): value for key, value in state['model'].items()}, strict=True)
    _, states = training_state_metadata()
    if state_sha256(state['model']) != states[190]['state_sha256']['ema_student']:
        raise ValueError('Saved checkpoint does not equal final EMA state fingerprint')
    return 190


def preflight(use_cuda=False, expected_gpus=1, phase="train"):
    freeze = verify_frozen()
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
    if phase == 'train':
        from train_data import CIFAR100TrainOnly
        dataset = CIFAR100TrainOnly(root='./data')
    else:
        dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False)
    expected_samples = 50000 if phase == 'train' else 10000
    if (len(dataset) != expected_samples or len(dataset.classes) != 100
            or set(dataset.targets) != set(range(100))):
        raise ValueError('Unexpected CIFAR-100 split, classes or labels')
    print('preflight_dataset=CIFAR100 classes=100 phase={} samples={}'.format(phase, expected_samples))
    if phase == 'eval':
        print('RUNTIME_SOURCE_FILES_JSON=' + json.dumps(freeze['source_hashes'], sort_keys=True))
        print('RUNTIME_SCRIPT_FILES_JSON=' + json.dumps(freeze['script_hashes'], sort_keys=True))
        print('RUNTIME_MANIFEST_SHA256=' + freeze['manifest_sha256'])
    identity()
    if use_cuda:
        if not torch.cuda.is_available() or torch.cuda.device_count() != expected_gpus:
            raise RuntimeError('Expected exactly {} Slurm-visible CUDA GPUs'.format(expected_gpus))
        for device in range(expected_gpus):
            if 'A800' not in torch.cuda.get_device_name(device):
                raise RuntimeError('Expected requested A800 GPU')
            probe = torch.empty(1, device='cuda:' + str(device))
            print('preflight_gpu={} name={}'.format(device, torch.cuda.get_device_name(device)))
            del probe
    elif torch.cuda.is_initialized():
        raise RuntimeError('CPU preparation unexpectedly initialized CUDA')
    print('PRECHECK_OK mode=' + ('cuda' if use_cuda else 'cpu'))


def job_log(log, job, phase):
    if not re.fullmatch(r'[0-9]+', str(job)):
        raise ValueError('Invalid job ID')
    expected = GROUP / 'logs' / ('{}_stdout_{}.log'.format(phase, job))
    if Path(log).resolve() != expected.resolve():
        raise ValueError('Unexpected job log path')
    return expected


def verify_log_sources(text, freeze, prefix=''):
    for label, field in [('SOURCE_FILES_JSON', 'source_hashes'), ('SCRIPT_FILES_JSON', 'script_hashes'),
                         ('MANIFEST_SHA256', 'manifest_sha256')]:
        rows = [line.split('=', 1)[1] for line in text.splitlines() if line.startswith(prefix + label + '=')]
        if len(rows) != 1 or (json.loads(rows[0]) if label.endswith('JSON') else rows[0]) != freeze[field]:
            raise ValueError('Runtime sources, scripts or manifest changed: ' + label)
    if ('PRECHECK_OK mode=cuda' not in text.splitlines()
            or not any(line.startswith('A800_RUNTIME_OK gpus=1 ') for line in text.splitlines())
            or 'Traceback (most recent call last)' in text):
        raise ValueError('Missing CUDA/A800 runtime checks or job exception')


def verify_slurm_logs(job, log, phase='train'):
    expected = job_log(log, job, phase)
    stdout = GROUP / 'logs/slurm' / ('{}_{}.out'.format(phase, job))
    stderr = GROUP / 'logs/slurm' / ('{}_{}.err'.format(phase, job))
    if stdout.read_bytes() != expected.read_bytes() or stderr.read_bytes():
        raise ValueError('Slurm stdout disagrees with log or stderr is not empty')


def verify_state_log(text, metadata):
    for epoch in ('120', '158', '190'):
        marker = 'TRAINING_STATE epoch={} fingerprint_sha256={}'.format(
            epoch, metadata['training_fingerprint_sha256'][epoch])
        if text.splitlines().count(marker) != 1:
            raise ValueError('Missing unique training state log evidence at epoch ' + epoch)


def training_marker(completion):
    return ('TRAIN_COMPLETE variant={variant} checkpoint={checkpoint} checkpoint_sha256={checkpoint_sha256} '
            'epoch=190 training_seed={training_seed} error_swap={error_swap} clean_method={clean_method} '
            'selection_protocol={selection_protocol}').format(**completion)


def activation_markers(entry):
    return {'error_swap_activation': 'ERROR_SWAP_ACTIVE epoch=121 enabled=False',
            'clean_method_activation': 'CLEAN_METHOD_SCHEDULE epoch=121 method=' + entry['method'],
            'ce_transfer_activation': 'CE_TRANSFER_SCHEDULE epoch=159 method=' + entry['method']}


def verify_activations(text, entry):
    markers = activation_markers(entry)
    if any(text.splitlines().count(marker) != 1 for marker in markers.values()):
        raise ValueError('Missing unique clean-method activation marker')
    return markers


def start_training():
    freeze = verify_frozen()
    _, checkpoint = identity()
    if checkpoint.parent.exists():
        raise ValueError('Refusing existing training prefix: ' + str(checkpoint.parent))
    print('SOURCE_FILES_JSON=' + json.dumps(freeze['source_hashes'], sort_keys=True))
    print('SCRIPT_FILES_JSON=' + json.dumps(freeze['script_hashes'], sort_keys=True))
    print('MANIFEST_SHA256=' + freeze['manifest_sha256'])


def finish_training(log, job):
    freeze = verify_frozen()
    variant, checkpoint = identity()
    _, entry, _ = settings()
    text = job_log(log, job, 'train').read_text()
    verify_log_sources(text, freeze)
    activations = verify_activations(text, entry)
    epoch = checkpoint_epoch(checkpoint, variant)
    state_metadata, _ = training_state_metadata()
    verify_state_log(text, state_metadata)
    completion = dict(freeze, **state_metadata, variant=variant, experiment_id=entry['id'], job_id=str(job),
                      checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha256(checkpoint), epoch=epoch,
                      training_seed=entry['training_seed'], error_swap=entry['error_swap'], clean_method=entry['method'],
                      selection_protocol=SELECTION, checkpoint_role='ema_student',
                      training_log=str(Path(log).resolve()), **activations)
    destination = checkpoint.parent / 'training_complete.json'
    with destination.open('x') as stream:
        json.dump(completion, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(training_marker(completion))
    return completion


def verify_training(check_slurm=True):
    freeze = verify_frozen()
    variant, checkpoint = identity()
    _, entry, _ = settings()
    completion = json.loads((checkpoint.parent / 'training_complete.json').read_text())
    state_metadata, _ = training_state_metadata()
    expected = dict(freeze, **state_metadata, variant=variant, experiment_id=entry['id'],
                    checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha256(checkpoint), epoch=190,
                    training_seed=entry['training_seed'], error_swap=entry['error_swap'], clean_method=entry['method'],
                    selection_protocol=SELECTION, checkpoint_role='ema_student', **activation_markers(entry))
    if any(type(completion[key]) is not type(value) or completion[key] != value for key, value in expected.items()):
        raise ValueError('Completed training identity, checkpoint, states or frozen files changed')
    log = job_log(completion['training_log'], completion['job_id'], 'train')
    text = log.read_text()
    verify_log_sources(text, freeze)
    verify_activations(text, entry)
    if text.splitlines().count(training_marker(completion)) != 1:
        raise ValueError('Missing training completion marker')
    verify_state_log(text, state_metadata)
    verify_slurm_logs(completion['job_id'], log)
    if check_slurm:
        require_slurm_success(completion['job_id'])
    checkpoint_epoch(checkpoint, variant)
    print('TRAIN_COMPLETION_VERIFIED job=' + completion['job_id'])
    return completion


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['preflight', 'start', 'finish', 'verify'])
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--expected-gpus', type=int, choices=[1], default=1)
    parser.add_argument('--phase', choices=['train', 'eval'], default='train')
    parser.add_argument('--log')
    parser.add_argument('--job')
    args = parser.parse_args()
    if args.action == 'preflight':
        preflight(args.cuda, args.expected_gpus, args.phase)
    elif args.action == 'start':
        start_training()
    elif args.action == 'finish':
        finish_training(args.log, args.job)
    else:
        verify_training()

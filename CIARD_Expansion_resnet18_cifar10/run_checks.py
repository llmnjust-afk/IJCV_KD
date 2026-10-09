"""CPU preflight and frozen fixed-epoch identities; no automatic GPU execution."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

EXCLUDED = {'data', 'models', 'model', 'logs', '__pycache__', 'frozen_source'}
STATE_KEYS = {'student', 'ema_student', 'teacher', 'optimizer', 'teacher_optimizer', 'dynamics', 'rng'}
RECIPE_WEIGHTS = {'C': (1.0, 1.0), 'S25': (.25, .25), 'D50': (1.0, .5), 'D25': (1.0, .25)}


def _check_entry(entry):
    if (not isinstance(entry.get('recipe'), str) or entry['recipe'] not in RECIPE_WEIGHTS
            or not isinstance(entry.get('config'), dict)
            or type(entry.get('seed')) is not int or entry['seed'] not in (0, 1)
            or type(entry.get('final_epoch')) is not int or entry['final_epoch'] != 252
            or entry.get('selection_protocol') != 'fixed_epoch252_no_test_selection'
            or entry.get('state_epochs') != [120, 200, 240, 252]
            or any(type(epoch) is not int for epoch in entry['state_epochs'])):
        raise ValueError('Invalid prepared experiment schema')
    initial, final = RECIPE_WEIGHTS[entry['recipe']]
    expected = {'awp_nat_weight': initial, 'awp_nat_weight_final': final,
                'awp_nat_schedule_start': 200, 'awp_nat_schedule_warmup': 40}
    if any(entry.get(key) != value or entry['config'].get(key) != value
           for key, value in expected.items()):
        raise ValueError('Prepared recipe and AWP schedule disagree')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _hashes(pattern):
    return {str(path): sha256(path) for path in sorted(Path('.').rglob(pattern))
            if not EXCLUDED.intersection(path.parts)}


def source_hashes():
    return _hashes('*.py')


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


def prepared_entry():
    manifest = Path.cwd().parent / 'experiment_manifest.json'
    anchor = manifest.parent / 'preparation' / 'experiment_manifest.sha256'
    expected = anchor.read_text().strip().split()[0]
    if not re.fullmatch(r'[0-9a-f]{64}', expected) or sha256(manifest) != expected:
        raise ValueError('Prepared manifest SHA256 mismatch')
    entries = json.loads(manifest.read_text())['experiments']
    matches = [entry for entry in entries if entry['name'] == Path.cwd().name]
    if len(matches) != 1:
        raise ValueError('Expected one manifest identity for current directory')
    entry = matches[0]
    _check_entry(entry)
    if source_hashes() != entry['prepared_python_sha256']:
        raise ValueError('Prepared Python sources changed')
    if _hashes('*.sbatch') != entry['prepared_script_sha256']:
        raise ValueError('Prepared Slurm scripts changed')
    return entry, expected


def identity():
    entry, _ = prepared_entry()
    train, evaluation = literals('CIARD.py'), literals('attack_eval.py')
    expected = {'prefix': entry['prefix'], 'VARIANT_NAME': entry['name'],
                'RECIPE': entry['recipe'], 'STATE_EPOCHS': entry['state_epochs'],
                'TRAIN_SEED': entry['seed'], 'FINAL_EPOCH': entry['final_epoch'],
                'epochs': entry['final_epoch'], 'SELECTION_PROTOCOL': entry['selection_protocol'],
                'CFG': entry['config']}
    if any(train.get(key) != value for key, value in expected.items()):
        raise ValueError('Training literals disagree with prepared manifest')
    checkpoint = Path('model') / entry['prefix'] / ('student_epoch{}.pth'.format(entry['final_epoch']))
    if (checkpoint.resolve() != Path(entry['checkpoint']).resolve()
            or checkpoint.resolve() != Path(evaluation['path']).resolve()):
        raise ValueError('Training, manifest and evaluator checkpoint disagree')
    if evaluation['variant_name'] != entry['name'] or entry['student'] != 'ResNet18':
        raise ValueError('Training and evaluation identities disagree')
    return entry['name'], checkpoint


def require_slurm_success(job):
    if not re.fullmatch(r'[0-9]+', str(job)):
        raise ValueError('Invalid Slurm job ID')
    result = subprocess.run(['sacct', '-j', str(job), '--format=JobIDRaw,State,ExitCode', '-P', '-n'],
                            check=True, text=True, stdout=subprocess.PIPE)
    rows = [row.split('|') for row in result.stdout.splitlines()]
    if not any(row[:3] == [str(job), 'COMPLETED', '0:0'] for row in rows):
        raise ValueError('Slurm has not confirmed COMPLETED / 0:0 for job ' + str(job))


def _training_states(checkpoint, entry):
    from training_state import state_sha256
    from awp_consistency import awp_natural_weight
    records = {}
    for epoch in entry['state_epochs']:
        path = checkpoint.parent / ('training_state_{}.json'.format(epoch))
        record = json.loads(path.read_text())
        if not isinstance(record, dict):
            raise ValueError('Invalid training state schema: ' + str(path))
        expected = {'epoch': epoch, 'variant': entry['name'], 'training_seed': entry['seed'],
                    'selection_protocol': entry['selection_protocol'],
                    'recipe': entry['recipe'],
                    'awp_nat_weight': awp_natural_weight(entry['config'], epoch)}
        if (type(record.get('epoch')) is not int or type(record.get('training_seed')) is not int
                or type(record.get('awp_nat_weight')) not in (int, float)
                or any(record.get(key) != value for key, value in expected.items())):
            raise ValueError('Training state identity mismatch: ' + str(path))
        hashes = record.get('state_sha256')
        if (not isinstance(hashes, dict) or set(hashes) != STATE_KEYS
                or any(not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value)
                       for value in hashes.values())):
            raise ValueError('Incomplete training state hashes: ' + str(path))
        if state_sha256({'epoch': epoch, 'state_sha256': hashes}) != record.get('fingerprint_sha256'):
            raise ValueError('Training state fingerprint mismatch: ' + str(path))
        records[epoch] = record
    return records


def checkpoint_epoch(checkpoint, variant):
    import torch
    from cifar10_models import resnet18
    from training_state import state_sha256
    from awp_consistency import awp_natural_weight
    entry, _ = prepared_entry()
    if variant != entry['name'] or Path(checkpoint).resolve() != Path(entry['checkpoint']).resolve():
        raise ValueError('Unexpected checkpoint identity')
    state = torch.load(checkpoint, map_location='cpu')
    expected = {'epoch': entry['final_epoch'], 'variant': variant, 'training_seed': entry['seed'],
                'checkpoint_role': 'ema_student', 'selection_protocol': entry['selection_protocol'],
                'recipe': entry['recipe'],
                'awp_nat_weight': awp_natural_weight(entry['config'], entry['final_epoch']),
                'config': entry['config']}
    if (type(state.get('epoch')) is not int or type(state.get('training_seed')) is not int
            or type(state.get('awp_nat_weight')) not in (int, float)
            or any(state.get(key) != value for key, value in expected.items())):
        raise ValueError('Fixed EMA checkpoint metadata mismatch')
    model = resnet18()
    model.load_state_dict({key.replace('module.', ''): value for key, value in state['model'].items()}, strict=True)
    final = _training_states(Path(checkpoint), entry)[entry['final_epoch']]
    if (state.get('final_state_fingerprint') != final['fingerprint_sha256']
            or state_sha256(state['model']) != final['state_sha256']['ema_student']):
        raise ValueError('Checkpoint does not match final EMA training state')
    return state['epoch']


def _check_dataset(phase, torchvision):
    if phase not in ('train', 'eval'):
        raise ValueError('Expected preflight phase train or eval')
    # torchvision may hash test files for integrity; train=True only loads train images.
    train = torchvision.datasets.CIFAR10(root='./data', train=True, download=False)
    if len(train) != 50000:
        raise ValueError('Unexpected CIFAR-10 training split size')
    if phase == 'eval':
        test = torchvision.datasets.CIFAR10(root='./data', train=False, download=False)
        if len(test) != 10000:
            raise ValueError('Unexpected CIFAR-10 test split size')
    print('preflight_dataset_samples=train:50000' + (' test:10000' if phase == 'eval' else ''))


def preflight(use_cuda=False, phase='train'):
    identity()
    import torch
    import torchvision
    import loguru, numpy, torchattacks, autoattack
    from cifar10_models import wideresnet
    from cifar10_nat_teacher_models import cifar10_resnet56
    from awp_consistency import validate_config, method_enabled
    settings = literals('CIARD.py')
    cfg = settings['CFG']
    validate_config(cfg)
    if method_enabled(cfg, cfg['method_start'] + 1) and not settings['USE_CIARDPP']:
        raise ValueError('0909 methods require USE_CIARDPP=True')
    if torch.__version__ != '1.10.0+cu113' or torchvision.__version__ != '0.11.1+cu113':
        raise RuntimeError('Expected torch 1.10.0+cu113 / torchvision 0.11.1+cu113')
    print('preflight_torch={} torchvision={} cuda_build={}'.format(torch.__version__, torchvision.__version__, torch.version.cuda))
    specifications = [
        ('models/model_cifar_wrn.pt', '2ede52bd042bbdf40a0c27e8008034afd9cbb0b256b9077a255e555d25f957f4',
         wideresnet, 'fc.weight', (10, 640), 'WRN-34-10'),
        ('models/nat_teacher_checkpoint/cifar10_resnnet56.pth',
         '9e1d3395f0a8c34296ca8cd4875b9b5177d53f79e89af9b88e1a6724c6d6c860',
         cifar10_resnet56, 'fc.weight', (10, 64), 'ResNet-56')]
    for path, digest, factory, key, shape, architecture in specifications:
        if sha256(path) != digest:
            raise ValueError('Teacher SHA256 mismatch: ' + path)
        state = torch.load(path, map_location='cpu')
        state = {key.replace('module.', ''): value for key, value in state.items()}
        if tuple(state[key].shape) != shape or any(k.endswith(('mu', 'sigma')) for k in state):
            raise ValueError('Unexpected teacher architecture / normalization')
        model = factory()
        model.load_state_dict(state, strict=True)
        print('teacher={} checkpoint={} bytes={} sha256={} shape={} input=raw'.format(
            architecture, Path(path).resolve(), Path(path).stat().st_size, digest, shape))
        del model, state
    _check_dataset(phase, torchvision)
    if phase == 'eval':
        variant, checkpoint = identity()
        checkpoint_epoch(checkpoint, variant)
    if use_cuda:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Expected exactly one Slurm-visible CUDA GPU')
        if '4090' not in torch.cuda.get_device_name(0):
            raise RuntimeError('Expected requested 4090 GPU')
        probe = torch.empty(1, device='cuda:0')
        print('preflight_gpu_name=' + torch.cuda.get_device_name(0))
        del probe
    elif torch.cuda.is_initialized():
        raise RuntimeError('CPU preparation unexpectedly initialized CUDA')
    print('PRECHECK_OK mode={} phase={}'.format('cuda' if use_cuda else 'cpu', phase))


def start_training():
    _, checkpoint = identity()
    if checkpoint.parent.exists():
        raise ValueError('Refusing existing training prefix: ' + str(checkpoint.parent))
    entry, manifest_hash = prepared_entry()
    print('SOURCE_FILES_JSON=' + json.dumps(source_hashes(), sort_keys=True))
    print('RUN_PREPARATION_JSON=' + json.dumps({'manifest_sha256': manifest_hash,
          'script_sha256': entry['prepared_script_sha256']}, sort_keys=True))


def _check_training_log(text, entry, manifest_hash):
    def record(prefix):
        rows = [line[len(prefix):] for line in text.splitlines() if line.startswith(prefix)]
        if len(rows) != 1:
            raise ValueError('Missing or duplicate log record: ' + prefix)
        return json.loads(rows[0])
    if record('SOURCE_FILES_JSON=') != source_hashes():
        raise ValueError('Source files changed during training')
    if record('RUN_PREPARATION_JSON=') != {'manifest_sha256': manifest_hash,
                                          'script_sha256': entry['prepared_script_sha256']}:
        raise ValueError('Preparation identity changed during training')
    lines = text.splitlines()
    finished = 'TRAINING_FINISHED epoch={} variant={}'.format(entry['final_epoch'], entry['name'])
    if (lines.count('PRECHECK_OK mode=cuda phase=train') != 1 or lines.count(finished) != 1
            or 'Traceback (most recent call last)' in text):
        raise ValueError('Training log lacks unique CUDA preflight/final epoch marker or contains an exception')


def finish_training(log, job):
    if not re.fullmatch(r'[0-9]+', str(job)):
        raise ValueError('Invalid training job ID')
    variant, checkpoint = identity()
    entry, manifest_hash = prepared_entry()
    raw = Path(log).read_bytes()
    _check_training_log(raw.decode(), entry, manifest_hash)
    epoch = checkpoint_epoch(checkpoint, variant)
    state_files = {str(path): sha256(path) for path in
                   (checkpoint.parent / ('training_state_{}.json'.format(value))
                    for value in entry['state_epochs'])}
    completion = {'variant': variant, 'job_id': str(job), 'checkpoint': str(checkpoint.resolve()),
        'checkpoint_sha256': sha256(checkpoint), 'epoch': epoch, 'final_epoch': epoch,
        'training_seed': entry['seed'], 'selection_protocol': entry['selection_protocol'],
        'recipe': entry['recipe'],
        'source_hashes': source_hashes(), 'manifest_sha256': manifest_hash,
        'prepared_script_sha256': entry['prepared_script_sha256'], 'training_state_sha256': state_files,
        'training_log': str(Path(log).resolve()), 'training_log_prefix_bytes': len(raw),
        'training_log_prefix_sha256': hashlib.sha256(raw).hexdigest()}
    with (checkpoint.parent / 'training_complete.json').open('x') as stream:
        json.dump(completion, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('TRAIN_COMPLETE variant={} checkpoint={} checkpoint_sha256={}'.format(
        variant, checkpoint.resolve(), completion['checkpoint_sha256']))


def verify_training(check_slurm=True):
    variant, checkpoint = identity()
    entry, manifest_hash = prepared_entry()
    completion = json.loads((checkpoint.parent / 'training_complete.json').read_text())
    expected = {'variant': variant, 'checkpoint': str(checkpoint.resolve()),
                'checkpoint_sha256': sha256(checkpoint), 'source_hashes': source_hashes(),
                'manifest_sha256': manifest_hash, 'prepared_script_sha256': entry['prepared_script_sha256'],
                'epoch': entry['final_epoch'], 'final_epoch': entry['final_epoch'],
                'training_seed': entry['seed'], 'selection_protocol': entry['selection_protocol'],
                'recipe': entry['recipe']}
    if any(completion.get(key) != value for key, value in expected.items()):
        raise ValueError('Completed training identity, checkpoint or preparation no longer match')
    expected_states = {str(checkpoint.parent / ('training_state_{}.json'.format(value)))
                       for value in entry['state_epochs']}
    if (set(completion['training_state_sha256']) != expected_states
            or any(sha256(path) != value for path, value in completion['training_state_sha256'].items())):
        raise ValueError('Completed training state files changed')
    raw = Path(completion['training_log']).read_bytes()
    if hashlib.sha256(raw[:completion['training_log_prefix_bytes']]).hexdigest() != completion['training_log_prefix_sha256']:
        raise ValueError('Completed training log prefix changed')
    text = raw.decode()
    _check_training_log(text, entry, manifest_hash)
    marker = 'TRAIN_COMPLETE variant={} checkpoint={} checkpoint_sha256={}'.format(
        variant, checkpoint.resolve(), completion['checkpoint_sha256'])
    if text.splitlines().count(marker) != 1:
        raise ValueError('Missing or duplicate final training completion marker')
    if check_slurm:
        require_slurm_success(completion['job_id'])
    if checkpoint_epoch(checkpoint, variant) != completion['epoch']:
        raise ValueError('Fixed checkpoint epoch changed')
    print('TRAIN_COMPLETION_VERIFIED job=' + completion['job_id'])
    return completion


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['preflight', 'start', 'finish', 'verify'])
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--phase', choices=['train', 'eval'], default='train')
    parser.add_argument('--log')
    parser.add_argument('--job')
    args = parser.parse_args()
    if args.action == 'preflight':
        preflight(args.cuda, args.phase)
    elif args.action == 'start':
        start_training()
    elif args.action == 'finish':
        finish_training(args.log, args.job)
    else:
        verify_training()

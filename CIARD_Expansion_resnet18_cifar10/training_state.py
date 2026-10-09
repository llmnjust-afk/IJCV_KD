"""Hash training state without model forwards or random draws."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def state_sha256(value):
    digest = hashlib.sha256()

    def token(data):
        digest.update(str(len(data)).encode('ascii') + b':' + data)

    def visit(item):
        token(type(item).__name__.encode('utf-8'))
        if torch.is_tensor(item):
            token(str(item.dtype).encode('ascii'))
            token(repr(tuple(item.shape)).encode('ascii'))
            token(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, np.ndarray):
            token(item.dtype.str.encode('ascii'))
            token(repr(item.shape).encode('ascii'))
            token(item.tobytes(order='C'))
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif item is None or isinstance(item, (bool, int, float, str, np.generic)):
            token(repr(item).encode('utf-8'))
        else:
            raise TypeError('Unsupported state type: ' + type(item).__name__)
        token(b'end')

    visit(value)
    return digest.hexdigest()


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def record_training_state(path, epoch, variant, training_seed, student,
                          ema_student, teacher, optimizer, teacher_optimizer,
                          dynamics, *, selection_protocol, recipe, awp_nat_weight):
    states = {'student': student.state_dict(), 'ema_student': ema_student.state_dict(),
              'teacher': teacher.state_dict(), 'optimizer': optimizer.state_dict(),
              'teacher_optimizer': teacher_optimizer.state_dict(),
              'dynamics': dynamics, 'rng': rng_state()}
    hashes = {name: state_sha256(value) for name, value in states.items()}
    fingerprint = state_sha256({'epoch': epoch, 'state_sha256': hashes})
    record = {'epoch': epoch, 'variant': variant, 'training_seed': training_seed,
              'selection_protocol': selection_protocol,
              'recipe': recipe, 'awp_nat_weight': awp_nat_weight,
              'state_sha256': hashes, 'fingerprint_sha256': fingerprint}
    with Path(path).open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('TRAINING_STATE epoch={} fingerprint_sha256={}'.format(epoch, fingerprint), flush=True)
    return record

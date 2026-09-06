"""Require a completed training log for this exact checkpoint and source copy."""
from pathlib import Path
import sys

from check_eval_log import sha256


def verify_training(checkpoint, variant):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError('Missing or empty best checkpoint')
    digest = sha256(checkpoint)
    required = {
        'TRAIN_COMPLETE variant={} checkpoint={} checkpoint_sha256={}'.format(
            variant, checkpoint.resolve(), digest),
        'source_ciard_sha256=' + sha256('CIARD.py'),
        'split_target_mix_sha256=' + sha256('split_target_mix.py'),
        'loss_sha256=' + sha256('mtard_loss.py'),
    }
    for log in sorted(Path('logs').glob('train_stdout_*.log')):
        if required.issubset(set(log.read_text().splitlines())):
            print('TRAIN_COMPLETION_VERIFIED log={} checkpoint_sha256={}'.format(log.resolve(), digest))
            return
    raise ValueError('No TRAIN_COMPLETE log matches this checkpoint, variant and source hashes')


if __name__ == '__main__':
    verify_training(*sys.argv[1:])

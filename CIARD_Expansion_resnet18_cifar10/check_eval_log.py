"""Validate complete historical evaluator output without changing its attacks."""
import hashlib
import json
import math
from pathlib import Path
import re
import sys


def parse_metrics(text):
    if 'Traceback (most recent call last)' in text:
        raise ValueError('Evaluation contains a traceback')
    if text.count('white box attack') != 1 or text.count('blackbox attack') != 1:
        raise ValueError('Expected one white-box and one black-box section')
    before_black, black = text.split('blackbox attack')
    before_white, white = before_black.split('white box attack')

    def value(section, pattern, scale):
        matches = re.findall(pattern, section, flags=re.MULTILINE)
        if len(matches) != 1:
            raise ValueError('Missing or duplicate metric: ' + pattern)
        number = float(matches[0]) * scale
        if not math.isfinite(number) or not 0 <= number <= 100:
            raise ValueError('Invalid accuracy: ' + matches[0])
        return number

    number = r'([0-9]+(?:\.[0-9]+)?)'
    out = {'autoattack': value(before_white, r'^robust accuracy: ' + number + r'%\s*$', 1),
           'clean': value(white, r'student clean acc:\s*' + number + r'\s*$', 100)}
    for name, label in [('fgsm', 'FGSM Attack'), ('pgdsat', 'PGD_sat Attack'),
                        ('pgdtrades', 'PGD_trades Attack'), ('cw', 'CW L_inf')]:
        out['whitebox_' + name] = value(white, r'student robust acc under ' + label + r'\s+' + number + r'\s*$', 100)
    for name, label in [('pgdtrades', 'PGD_trades Attack'), ('square', 'Square Attack'), ('cw', 'CW L_inf')]:
        out['blackbox_' + name] = value(black, r'student robust acc under ' + label + r'\s+' + number + r'\s*$', 100)
    return out


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    log, checkpoint, expected_checkpoint, expected_evaluator, variant, job = sys.argv[1:]
    metrics = parse_metrics(Path(log).read_text())
    if sha256(checkpoint) != expected_checkpoint or sha256('attack_eval.py') != expected_evaluator:
        raise ValueError('Checkpoint or evaluator changed during evaluation')
    result = {
        'variant': variant, 'job_id': job, 'checkpoint': str(Path(checkpoint).resolve()),
        'checkpoint_sha256': expected_checkpoint, 'evaluator_sha256': expected_evaluator,
        'protocol': 'best_backup historical protocol', 'attack_seed': None,
        'selection_protocol': 'historical_50k_train_test_loader_selection',
        'metrics_percent': metrics,
        'note': 'Wrapper-parsed complete metrics; stochastic attacks are not explicitly seeded.',
    }
    destination = Path(checkpoint).parent / ('eval_best_0906v1_' + job + '.json')
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('EVAL_METRICS_COMPLETE result={}'.format(destination.resolve()))


if __name__ == '__main__':
    main()

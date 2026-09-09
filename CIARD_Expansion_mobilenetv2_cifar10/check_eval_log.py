"""Read the unchanged historical metric lines and append a percentage summary."""
import json
import math
from pathlib import Path
import re
import sys

from run_checks import sha256, verify_training

METRICS = [
    ('clean', 'Clean'), ('whitebox_fgsm', '白盒 FGSM'), ('whitebox_pgdsat', '白盒 PGDsat'),
    ('whitebox_pgdtrades', '白盒 PGDtrades'), ('whitebox_cw', '白盒 CW'),
    ('blackbox_pgdtrades', '黑盒 PGDtrades'), ('blackbox_square', '黑盒 Square'),
    ('blackbox_cw', '黑盒 CW'), ('autoattack', 'AA')]


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


def main():
    log, job = sys.argv[1:]
    metrics = parse_metrics(Path(log).read_text())
    completed = verify_training()
    result = {'variant': completed['variant'], 'job_id': str(job),
        'training_job_id': completed['job_id'], 'epoch': completed['epoch'], 'checkpoint': completed['checkpoint'],
        'checkpoint_sha256': completed['checkpoint_sha256'], 'evaluator_sha256': sha256('attack_eval.py'),
        'source_hashes': completed['source_hashes'], 'test_samples': 10000, 'attack_seed': None,
        'selection_protocol': 'historical_50k_train_test_loader_selection',
        'protocol': 'frozen_source_historical_attacks', 'metrics_percent': metrics,
        'correct_counts': {key: round(value * 100) for key, value in metrics.items()},
        'evaluation_log': str(Path(log).resolve()),
        'note': 'Test-loader selection bias; historical stochastic attacks not explicitly seeded.'}
    destination = Path(completed['checkpoint']).parent / ('eval_best_0909v1_' + str(job) + '.json')
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('========== 完整 10000 张测试集成绩（百分比） ==========')
    for key, label in METRICS:
        print('{}: {:.2f}%'.format(label, metrics[key]))
    print('八项均值: {:.2f}%'.format(sum(metrics[key] for key, _ in METRICS[:8]) / 8))
    print('EVAL_COMPLETE variant={} checkpoint_sha256={} evaluator_sha256={} result={}'.format(
        result['variant'], result['checkpoint_sha256'], result['evaluator_sha256'], destination.resolve()))


if __name__ == '__main__':
    main()

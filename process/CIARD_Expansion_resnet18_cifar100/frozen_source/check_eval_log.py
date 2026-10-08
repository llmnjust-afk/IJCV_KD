"""Read the unchanged historical metric lines and append a percentage summary."""
import json
import math
from pathlib import Path
import re
import sys


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



def verify_evaluation_log(text, completed):
    from run_checks import verify_log_sources
    verify_log_sources(text, completed, prefix='RUNTIME_')
    if 'preflight_dataset=CIFAR100 classes=100 phase=eval samples=10000' not in text.splitlines():
        raise ValueError('Missing full10000 test dataset evidence')
    actual = re.findall(r'^EVAL_SEED metric=(\w+) seed=0 cudnn_deterministic=True benchmark=False$',
                        text, flags=re.MULTILINE)
    if len(actual) != 9 or set(actual) != {key for key, _ in METRICS}:
        raise ValueError('Every metric must explicitly reset the paired seed')
    return parse_metrics(text)


def finish_evaluation(log_name, job):
    from run_checks import GROUP, settings, verify_training, sha256, job_log, PROTOCOL, SELECTION
    log = job_log(log_name, job, 'eval')
    completed = verify_training()
    metrics = verify_evaluation_log(log.read_text(), completed)
    _, entry, _ = settings()
    counts = {key: round(value * 100) for key, value in metrics.items()}
    if any(abs(metrics[key] - count / 100) > 1e-8 for key, count in counts.items()):
        raise ValueError('Metrics do not correspond to integer full10000 correct counts')
    fields = ('variant', 'experiment_id', 'epoch', 'checkpoint', 'checkpoint_sha256', 'training_seed',
              'error_swap', 'clean_method', 'selection_protocol', 'checkpoint_role', 'source_hashes', 'script_hashes',
              'manifest_sha256', 'training_state_sha256', 'training_fingerprint_sha256', 'error_swap_activation',
              'clean_method_activation', 'ce_transfer_activation')
    result = {key: completed[key] for key in fields}
    result.update(job_id=str(job), training_job_id=completed['job_id'], evaluator_sha256=sha256(GROUP / 'attack_eval.py'),
                  test_samples=10000, attack_seed=0, protocol=PROTOCOL, metrics_percent=metrics,
                  correct_counts=counts, evaluation_log=str(log.resolve()))
    destination = Path(completed['checkpoint']).parent / ('eval_' + str(job) + '.json')
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('========== 完整 10000 张测试集成绩（百分比） ==========')
    for key, label in METRICS:
        print('{}: {:.2f}%'.format(label, metrics[key]))
    print('七攻击均值: {:.2f}%'.format(sum(metrics[key] for key, _ in METRICS[1:8]) / 7))
    print('八项均值: {:.2f}%'.format(sum(metrics[key] for key, _ in METRICS[:8]) / 8))
    print('EVAL_COMPLETE variant={} checkpoint_sha256={} evaluator_sha256={} result={}'.format(
        entry['name'], result['checkpoint_sha256'], result['evaluator_sha256'], destination.resolve()))
    return result


if __name__ == '__main__':
    finish_evaluation(*sys.argv[1:])

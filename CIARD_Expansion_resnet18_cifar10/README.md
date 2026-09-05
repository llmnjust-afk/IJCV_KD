# CIFAR-10 / ResNet-18 0906 target-mixing candidate

Selected from `run/0906v1/resnet18_tmix_a020_s120_w40_p081740` on 2026-09-06.
The source experiment has started training, but no completed 0906 evaluation
was available at selection. This is an unvalidated candidate, not a new best
result. The verified ResNet reference remains the 0703 PCGrad mainline.

## Configuration and method

| Setting | Fixed value |
| --- | --- |
| target_mix_alpha | 0.20 |
| target_mix_start / target_mix_warmup | 120 / 40 epochs |
| push_lambda | 0.081740 |
| epochs / batch size / training seed | 300 / 128 / 0 |
| student / dataset | ResNet-18 / CIFAR-10 |
| prefix | `Cifar10_ResNet18_0906v1_tmix_a020_s120_w40_p081740` |
| best checkpoint | `model/Cifar10_ResNet18_0906v1_tmix_a020_s120_w40_p081740/student_best.pth` |
| Slurm resources | `rtx4090`, `aias-compute-2`, `gpu:4090:1` |

Only the adversarial KD target changes relative to the 0703 baseline. Let
`q_adv` be the existing robust soft target and `q_clean` the live robust teacher's
already-computed clean prediction, both detached and softened with the same
current-batch `temp_adv` before its update. With `m` indicating a correct clean
top-1 prediction, the target is:

```text
a_e = 0.20 * clamp((epoch - 120) / 40, 0, 1)
q_mix = (1 - a_e * m) * q_adv + a_e * m * q_clean
```

Mixing first becomes nonzero at epoch 121 and reaches full strength at epoch 160.
Zero mixing uses the original KL path; the original batch-and-class mean reduction
is preserved. No extra teacher forward, BN update, FGSM auxiliary branch, or CE
branch is added. Natural KD, temperature updates, push, teacher-margin PCGrad,
EMA, checkpoint selection, and weight averaging retain the baseline behavior.
The historical `teacher_margin_conflict_scale` is still only computed/logged,
not applied; its configured floor is not an explanation for any future gain.

All settings are literal values in `CIARD.py` and printed at startup. Mixing
strength, correct-clean fraction, effective weight, target L1 distance, and
temperature are logged every 100 steps. Do not edit source after submitting a
job. Repeated experiments need a new directory and prefix; training refuses an
existing prefix.

## Protocol and preparation

Training uses all 50,000 training images and the historical test loader to select
best, so results have test-selection bias. Only this run's `student_best` is the
primary evaluation checkpoint; weight-average outputs remain auxiliary.
The frozen historical evaluator does not explicitly seed every stochastic attack.
`check_eval_log.py` requires Clean, four white-box metrics, three black-box metrics,
and AutoAttack, and checks checkpoint/evaluator hashes before writing JSON with
`attack_seed: null`. This remains historical-compatible evaluation.

The target is one checkpoint strictly exceeding the eight CIARD baselines:
Clean `88.87`, FGSM `61.88`, PGDsat `51.70`, PGDtrades `54.46`, CW `50.61`,
black-box PGDtrades `66.28`, Square `80.03`, black-box CW `64.79`;
AutoAttack should additionally be at least `48.88`. No such result is claimed.

Use the existing `ciard` environment: Python 3.8.20, PyTorch 1.10.0+cu113,
torchvision 0.11.1+cu113; direct dependencies are in `requirements.txt`.
The wrappers load the site's CUDA 11.8 module; PyTorch's build uses CUDA 11.3.
Teachers use raw inputs without normalization:

| Teacher path | SHA256 |
| --- | --- |
| `models/model_cifar_wrn.pt` (WRN-34-10) | `2ede52bd042bbdf40a0c27e8008034afd9cbb0b256b9077a255e555d25f957f4` |
| `models/nat_teacher_checkpoint/cifar10_resnnet56.pth` (ResNet-56) | `9e1d3395f0a8c34296ca8cd4875b9b5177d53f79e89af9b88e1a6724c6d6c860` |

Local resources and outputs are not published. Before manual use on the original
cluster, prepare these from this directory (create links only if absent):

```bash
mkdir -p logs/slurm model
ln -s /home/lixidong25/mycode/CIARD_Expansion/data data
ln -s /home/lixidong25/mycode/CIARD_Expansion/models models
```

The wrappers' working and Slurm output paths point to
`/home/lixidong25/mycode/CIARD_Expansion/origin_code/0906v1/IJCV_KD/CIARD_Expansion_resnet18_cifar10`.
Adapt those site paths if cloning elsewhere. `train_4090.sbatch` and
`eval_4090_best.sbatch` are user-submitted only. Evaluation requires successful
training status, `TRAIN_COMPLETE`, and a nonempty best checkpoint; completion
requires successful evaluation status, all nine metrics, JSON, and `EVAL_COMPLETE`.
These wrappers use this copy's output directory, never the running experiment's
outputs. No training or evaluation is launched as part of publication.

## Source provenance

Python and dependency files are copied byte-for-byte from the selected experiment.
Only wrapper working/output paths and Slurm job names change; the prefix stays
the same within the separate output directory.

| File | SHA256 |
| --- | --- |
| CIARD.py | `c75e20992a487a7b458691e6c8f170e7eb7e27a4b918e5cddbe1bdf30dc9d382` |
| attack_eval.py | `ee28e4df5279ba8edeca847d5270188db2598279fb66524292533c8c94f75637` |
| mtard_loss.py | `07ec022c626fc4b1c87170a280ed0ef360d5c064a79f26b47ee258374da2528d` |

Publication checks passed: byte-for-byte source comparison, Python/Shell syntax,
CPU target-mixing loss/gradient/masking/ramp checks, baseline loop equivalence,
and the existing tiny-network smoke test. The log checker was tested against
complete historical logs and temporary incomplete/duplicate/hash-change fixtures.
These checks validate the copied implementation and wrappers, not model accuracy;
no GPU training or attack evaluation was run for publication.

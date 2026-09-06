# CIFAR-10 / ResNet-18 0906v2 split-mixing candidate

Selected from `run/0906v2/resnet18_split_t025_n020_s120_w40_p081740`.
This candidate is based on 0906v1 G3, whose aggregate result slightly exceeded
the previous best ResNet; see the [completed comparison](../README.md#completed-0906v1-resnet-result-versus-the-previous-best).
No completed 0906v2 evaluation was available at publication. Selection reflects
a conservative method configuration, not a performance ranking.

## Fixed configuration

| Setting | Value |
| --- | --- |
| split_target_mix | True |
| target_mix_alpha (G3 reference) | 0.20 |
| split_target_alpha | 0.25 |
| split_nontarget_alpha | 0.20 |
| target_mix_start / target_mix_warmup | 120 / 40 |
| push_lambda | 0.081740 |
| epochs / batch / training seed | 300 / 128 / 0 |
| prefix | `Cifar10_ResNet18_0906v2_split_t025_n020_s120_w40_p081740` |
| checkpoint | `model/Cifar10_ResNet18_0906v2_split_t025_n020_s120_w40_p081740/student_best.pth` |
| Slurm resources | `rtx4090`, `aias-compute-2`, `gpu:4090:1` |

Train from scratch; no 0906v1 student weights are loaded. Configuration is fixed
in the initial CFG and printed at startup. Do not edit source after submitting;
new runs require their own directory/prefix. Existing prefixes are refused.

## Method

Let q(a) be the original mixture of robust-teacher adversarial and clean targets,
using the clean-top1-correct mask and ramp clamp((epoch-120)/40,0,1). Both
teacher distributions are detached and use the current batch's temp_adv before
its update; the student's log-softmax retains temperature 1.

B(a) is KL on the true-class versus all-other-classes binary distribution.
N(a) is KL on the normalized non-target distribution. w is the total non-target
probability in the G3 reference q(0.20). Per sample, this candidate uses:

```text
L_new = L_G3 + [B(0.25) - B(0.20) + w * (N(0.20) - N(0.20))] / 10
```

The N correction is zero for this chosen variant: it retains G3's conditional
non-target distribution and its mass weight while changing binary supervision.
The shared helper also supports the independent non-target coefficients used
in the other prepared experiments. This follows the target/non-target separation
idea of [DKD](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html),
with a G3 reference weight and loss correction specific to this experiment.

The original log(q+1e-5) convention and batch-by-class mean are preserved.
Log-space mixtures handle saturated targets; incorrect-clean-teacher samples
retain their original terms. Mixing starts at epoch 121 and is full at 160.
There are no extra model forwards, teacher gradients, BN updates, or random
samples. Natural KD, teacher updates, push, teacher-margin PCGrad, EMA,
checkpoint selection and WA keep G3 behavior. The historical
teacher_margin_conflict_scale remains logged but unapplied.

Actual CFG is printed as a multiline block; every 100 steps, splitmix records
both effective alphas, binary/conditional KLs, reference_other_mass and the loss
correction. The full eight-run matrix remains in the local run/0906v2 directory.

## Protocol and preparation

Keep the historical 50k train/test-loader selection protocol and frozen evaluator.
Only this run's EMA student_best is the primary target; WA/sweep outputs are
auxiliary. Attack seeds are not fully fixed and JSON records attack_seed=null.
check_eval_log.py requires all nine metrics and unchanged checkpoint/evaluator
hashes before writing eval_best_0906v2_<jobid>.json and EVAL_COMPLETE.
Eight-metric all-improvement remains the goal, not an achieved result.

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
`/home/lixidong25/mycode/CIARD_Expansion/origin_code/0906v2/IJCV_KD/CIARD_Expansion_resnet18_cifar10`.
Adapt those site paths if cloning elsewhere. `train_4090.sbatch` and
`eval_4090_best.sbatch` are user-submitted only. Evaluation requires successful
training status, `TRAIN_COMPLETE`, and a nonempty best checkpoint.
`check_train_complete.py` additionally requires the exact variant, checkpoint
hash, and current CIARD/helper/loss source hashes to match the training log; completion
requires successful evaluation status, all nine metrics, JSON, and `EVAL_COMPLETE`.
These wrappers use this copy's output directory, never the running experiment's
outputs. No training or evaluation is launched as part of publication.

## Provenance and checks

All 18 Python files plus requirements.txt are byte-identical to the selected
experiment. Only the two wrappers' working/output paths and Slurm job names
are adapted to this publication copy. Data, weights, links and outputs are not
copied. The experiment directory and previous publication copy are untouched.

Publication checks passed: source-byte comparisons, Python and shell syntax,
CPU loss/gradient equivalence and gating checks, the actual training loss
fragment, and completion/hash checks with historical evaluation logs and
negative fixtures. These checks do not establish model performance.

| File | SHA256 |
| --- | --- |
| CIARD.py | `cae552713ddeb924302698296886f0971a5cb24270b214abcdd35683f9930778` |
| split_target_mix.py | `ef059f02861714aba63ff7ad4a4125c8d2823cde332cda0106067e2604e3fbf4` |
| attack_eval.py | `6a17008d5aa69609bac39a8a7bd2b9483477e415e41ab97470a4e891a8dae90c` |
| mtard_loss.py | `07ec022c626fc4b1c87170a280ed0ef360d5c064a79f26b47ee258374da2528d` |
| check_train_complete.py | `c8fecb28b3edc261245f6dffeda846c7345656c05ed587c59f2bc1939841d837` |

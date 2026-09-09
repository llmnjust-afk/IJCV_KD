# IJCV_KD — 0909v1 candidate sources

2026-09-09: the two active model entries now contain **ResNet R5** and **MobileNet M2**, copied from the independent 0909v1 experiments selected by the user. Both use two adversarial views, KD-AWP gamma **0.002**, and consistency weight **0.5**, with start120/warmup40 and JS temperature0.5. **Full evaluation is pending; these are candidate sources, not newly verified best models.**

| Entry | Source under `run/0909v1` | Preserved base recipe |
| --- | --- | --- |
| [ResNet-18 R5](CIARD_Expansion_resnet18_cifar10/README.md) | `resnet18_g7_v2_awp0p002_cr0p50` | G7 target=.25/nontarget=0, push=.081740, PCGrad, EMA |
| [MobileNet-V2 M2](CIARD_Expansion_mobilenetv2_cifar10/README.md) | `mobilenetv2_best_v2_awp0p002_cr0p50` | verified push=.05 tm010-repeat, original backpropagation, EMA |

All 40 copied Python files match their respective experiments byte for byte, including CFG, prefix, model definitions, losses, evaluator and completion checks. No architecture, teacher, data or evaluation protocol change was introduced by synchronization. [SYNC_MANIFEST.json](SYNC_MANIFEST.json) records the source hashes, script adaptations and verification summary. **Every file under `best_backup/` remains unchanged.**

## Candidate method and selection

The first 120 epochs retain the original training path. From epoch 121, each image has two independent original crop/flip views, each with PGD-10 at 8/255 and step 2/255. The full original losses are averaged over views. At epoch 160 the added terms reach their configured strength, with `r=clip((epoch-120)/40,0,1)`.

KD-AWP uses a proxy gradient of the original natural/adversarial distillation objective and applies `v=gamma*r*||w||*g/(||g||+1e-12)` to convolution/linear weights only. The actual full objective is differentiated at perturbed weights; weights are restored before SGD and EMA. Consistency adds `lambda*r*mean(JS(softmax(z_adv1/.5),softmax(z_adv2/.5)))/10`, without a temperature-squared multiplier. Teacher, student and dynamic-temperature updates still occur once per logical batch; ResNet retains its original margin PCGrad.

R5 represents the full proposed combination at a lower perturbation strength than R6's 0.005. M2 tests the same combination on MobileNet's verified base recipe. This is a configuration-based selection, not a ranking supported by completed results. Attribution: [Adversarial Weight Perturbation, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/1ef91c212e30e14bf125e9374262401f-Abstract.html) and [Consistency Regularization for Adversarial Robustness, AAAI 2022](https://arxiv.org/pdf/2103.04623). These are CIARD adaptations, not reproductions of the full published recipes or proof of a new contribution. Two views require extra computation.

## Execution and result status

The existing independent experiments remain the training/evaluation locations. Their source and Slurm scripts are unchanged by this synchronization; no duplicate training is needed. Package wrappers are source templates adapted to the two package directories, with ResNet on compute-4 and MobileNet on compute-2, each requesting one 4090. The package intentionally has no resource symlinks, datasets, checkpoints, outputs or logs, and cannot be submitted directly without preparing an independent run. Both entries use the minimal dependency list for the already verified ciard environment.

After user-submitted training succeeds, evaluate the fixed EMA `student_best.pth` in the corresponding run directory. Preserve the original eight metrics plus AutoAttack, and report M2−M1 separately. ResNet's target remains all eight metrics strictly above the CIARD reference and AA at least 48.88%. No complete R5/M2 test result exists at synchronization time. Full comparison and commands remain in the local batch README. Only the user submits GPU jobs.

## Historical results retained as references

The following completed results belong to earlier configurations, **not to R5 or M2**. The active ResNet entry previously used G7 alone; the active MobileNet entry previously used the 0903 push=.075 candidate. Old SARD scripts/configs and design documents remain historical material, not the current execution interface.

## Completed 0906v2 G7 result and backup

[ResNet-18 / CIFAR-10 0906v2 G7](best_backup/resnet18_cifar10_0906v2/README.md)
is now available as a separate lightweight backup, using target mixing 0.25,
non-target mixing 0.00 and reference mixing 0.20. **Seven of eight primary
metrics exceed the published CIARD baseline. FGSM is 61.30 versus 61.88,
leaving a 0.58 percentage-point gap.** AutoAttack is 49.25%.

The eight-metric mean is 65.01%, below this batch's G1 mean of 65.11%.
Selection reflects baseline coverage, not the highest mean or an improvement
on every metric. The linked Chinese README includes all eight comparisons,
configuration, source/checkpoint hashes and original evidence paths.
Only source, wrappers and documentation are archived; no weights or outputs
are uploaded. The active ResNet entry now adds the R5 mechanisms to G7; the
independent G7 backup and earlier backups are preserved. G7 uses the same
historical protocol limitations described below.

## Completed 0906v1 ResNet result versus the previous best

**0906v1 G3 slightly outperformed the previous best ResNet (0703 PCGrad
Optuna-transfer) on aggregate in this completed single-run evaluation.**
The eight-metric mean increased from **64.99 to 65.11** (+0.12 percentage
points), the seven-attack mean from **61.61 to 61.73** (+0.12), and AutoAttack
from **49.08 to 49.36** (+0.28).

Accuracies are percentages; changes are percentage points. The means exclude
AutoAttack and are arithmetic summaries, not joint worst-case robust accuracy.

| Metric | Previous best (0703) | 0906v1 G3 | Change |
| --- | ---: | ---: | ---: |
| Clean | 88.66 | 88.76 | +0.10 |
| White-box FGSM | 61.33 | 61.39 | +0.06 |
| White-box PGDsat | 52.21 | 52.03 | -0.18 |
| White-box PGDtrades | 54.76 | 54.66 | -0.10 |
| White-box CW | 51.27 | 51.60 | +0.33 |
| Black-box PGDtrades | 66.60 | 66.91 | +0.31 |
| Square | 80.09 | 80.10 | +0.01 |
| Black-box CW | 65.02 | 65.42 | +0.40 |
| AutoAttack (separate) | 49.08 | 49.36 | +0.28 |
| Seven-attack mean | 61.61 | 61.73 | +0.12 |
| Eight-metric mean | 64.99 | 65.11 | +0.12 |

Six of eight primary metrics improved, but PGDsat and PGDtrades decreased by
0.18 and 0.10 points. Against the published CIARD baseline, Clean 88.76 is
still below 88.87 and FGSM 61.39 below 61.88. Thus this is a small aggregate
improvement, **not an improvement on every metric**, and not a demonstrated
multi-seed gain. The frozen best-backup reference remains unchanged.

Evidence: 0906v1 variant `resnet18_tmix_a020_s120_w40_p081740`, training job
132014, evaluation job 132030, EMA `student_best.pth` checkpoint SHA256
`e18584ba3f84e936f9fbcb80caa4a4297e1df03998841cc256affc22a24d9b89`;
evaluator SHA256 `ee28e4df5279ba8edeca847d5270188db2598279fb66524292533c8c94f75637`.
The completed local result report is `结果分析/0906v1_结果分析.md`; the historical
0703 comparison is recorded in `LOCAL_VERSION_MANAGEMENT.md`. Those local
reports/weights are not bundled here. The preceding
[0906v1 source commit](https://github.com/llmnjust-afk/IJCV_KD/commit/b7c14e7d5ac87f2c66ca9c56aa95deb4dd222c1c)
preserves the evaluated candidate's source. These results belong to 0906v1,
not to G7 or the active 0909v1 R5/M2 candidates.

Both versions use the historical 50,000-image training protocol and test-loader
checkpoint selection, which introduces selection bias. Stochastic attack seeds
are not all explicitly fixed. PGDtrades uses step size 0.003 in the frozen
evaluator; the paper describes 2/255, so comparison to its published numbers
is not proof of an exactly matched reproduction.

## Other source and execution notes

The previous 0903 MobileNet push=.075 candidate missed the CIARD reference on Clean (89.07 versus 89.51) and black-box CW (65.29 versus 66.12). It has now been replaced in the active entry by M2, based on the verified 0624 push=.05 recipe. The verified reference and all earlier backups remain unchanged. Historical auxiliary MobileNet tools remain available, but the active path uses the copied 4090 wrappers and frozen full evaluator.

Both models retain raw-input WRN-34-10 robust and ResNet-56 natural teachers. Training remains 50,000 images with test-loader checkpoint selection, which introduces selection bias. Stochastic attack seeds are not all explicitly fixed; frozen PGDtrades uses step 0.003 while the paper describes 2/255. Earlier numerical results and future candidate results must retain these qualifications.

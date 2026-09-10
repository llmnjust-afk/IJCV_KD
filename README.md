# IJCV_KD — 0909v1 R5 evaluated source and M2 candidate

2026-09-10: the active entries contain **ResNet R5** and **MobileNet M2**, both using two adversarial views, KD-AWP gamma **0.002**, consistency weight **0.5**, start120/warmup40 and JS temperature0.5. **R5 has completed evaluation: seven of eight primary metrics exceed the paper baseline, with FGSM only 0.12 pp below. M2 has no complete evaluation result yet.** The user-selected R5 source is now preserved in [the 0909v1 backup](best_backup/resnet18_cifar10_0909v1/README.md).

| Entry | Source under `run/0909v1` | Preserved base recipe |
| --- | --- | --- |
| [ResNet-18 R5](CIARD_Expansion_resnet18_cifar10/README.md) | `resnet18_g7_v2_awp0p002_cr0p50` | G7 target=.25/nontarget=0, push=.081740, PCGrad, EMA |
| [MobileNet-V2 M2](CIARD_Expansion_mobilenetv2_cifar10/README.md) | `mobilenetv2_best_v2_awp0p002_cr0p50` | verified push=.05 tm010-repeat, original backpropagation, EMA |

All 40 copied Python files match their respective experiments byte for byte, including CFG, prefix, model definitions, losses, evaluator and completion checks. No architecture, teacher, data or evaluation protocol change was introduced by synchronization. [SYNC_MANIFEST.json](SYNC_MANIFEST.json) records the source hashes, script adaptations and verification summary. The three earlier backup directories remain byte-identical; the backup index now also lists the new R5 snapshot.

## Candidate method and selection

The first 120 epochs retain the original training path. From epoch 121, each image has two independent original crop/flip views, each with PGD-10 at 8/255 and step 2/255. The full original losses are averaged over views. At epoch 160 the added terms reach their configured strength, with `r=clip((epoch-120)/40,0,1)`.

KD-AWP uses a proxy gradient of the original natural/adversarial distillation objective and applies `v=gamma*r*||w||*g/(||g||+1e-12)` to convolution/linear weights only. The actual full objective is differentiated at perturbed weights; weights are restored before SGD and EMA. Consistency adds `lambda*r*mean(JS(softmax(z_adv1/.5),softmax(z_adv2/.5)))/10`, without a temperature-squared multiplier. Teacher, student and dynamic-temperature updates still occur once per logical batch; ResNet retains its original margin PCGrad.

R5 is the evaluated, user-selected ResNet source: all eight primary results improve over the same-batch G7 control R1. It retains seven improvements over the paper baseline but still misses FGSM by 0.12 pp. R6 has a higher eight-metric mean but three strong white-box results below the paper baseline. M2 tests the R5 combination on MobileNet's verified base recipe; its selection remains a method-transfer candidate pending full evaluation. Attribution: [Adversarial Weight Perturbation, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/1ef91c212e30e14bf125e9374262401f-Abstract.html) and [Consistency Regularization for Adversarial Robustness, AAAI 2022](https://arxiv.org/pdf/2103.04623). These are CIARD adaptations, not reproductions of the full published recipes or proof of a new contribution. Two views require extra computation.

## Execution and result status

The existing independent experiments remain the training/evaluation locations. Their source and Slurm scripts are unchanged by this synchronization; no duplicate training is needed. Package wrappers are source templates adapted to the two package directories, with ResNet on compute-4 and MobileNet on compute-2, each requesting one 4090. The package intentionally has no resource symlinks, datasets, checkpoints, outputs or logs, and cannot be submitted directly without preparing an independent run. Both entries use the minimal dependency list for the already verified ciard environment.

After user-submitted training succeeds, evaluate the fixed EMA `student_best.pth` in the corresponding run directory. Preserve the original eight metrics plus AutoAttack, and report M2−M1 separately. ResNet's target remains all eight metrics strictly above the CIARD reference and AA at least 48.88%. R5 training/evaluation jobs 132648/132743 completed successfully; the fixed EMA is epoch 230 and AutoAttack is 49.35%. M2 still has no complete test result. R5's results are below and in its backup README; original logs and manual commands remain in the local batch. This source publication requires no repeat training or evaluation. Only the user submits GPU jobs.

## Completed results versus the paper CIARD baseline

The tables contain completed **CIFAR-10** evaluations: historical G3/G7 and MobileNet references, plus the newly evaluated **0909v1 R5**. **M2 has no result in these tables.** Each configuration uses one selected `student_best.pth`; values are not combined across checkpoints.

Accuracies are percentages. Parentheses show **this result minus the paper CIARD baseline**, in percentage points (pp): `61.30 (-0.58)` means 0.58 pp below the baseline. The baseline is the published CIARD result for the **same student model**, not an earlier version of this repository. Seven-attack and eight-metric means are computed from the corresponding metric values; AutoAttack is separate. Means and differences are computed before rounding, so subtracting two displayed means can differ by 0.01 pp. These means are descriptive summaries, not joint worst-case accuracy or the paper's W-Robust metric.

Baseline sources: [CIARD supplementary material](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Lu_CIARD_Cyclic_Iterative_ICCV_2025_supplemental.pdf), Tables 1/2 for Clean and white-box ResNet/MobileNet results, and Table 5 for ResNet AutoAttack; [CIARD paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Lu_CIARD_Cyclic_Iterative_Adversarial_Robustness_Distillation_ICCV_2025_paper.pdf), Tables 5/6, **Robust** columns, for black-box ResNet/MobileNet results. The local copies of these PDFs were used for verification.

### ResNet-18 / CIFAR-10

| Metric | Paper CIARD baseline | 0906v1 G3 (Δ vs baseline) | 0906v2 G7 (Δ vs baseline) | 0909v1 R5 (Δ vs baseline) |
| --- | ---: | ---: | ---: | ---: |
| Clean | 88.87 | 88.76 (-0.11) | 88.91 (+0.04) | 89.07 (+0.20) |
| White-box FGSM | 61.88 | 61.39 (-0.49) | 61.30 (-0.58) | 61.76 (-0.12) |
| White-box PGDsat | 51.70 | 52.03 (+0.33) | 51.91 (+0.21) | 52.34 (+0.64) |
| White-box PGDtrades | 54.46 | 54.66 (+0.20) | 54.54 (+0.08) | 55.01 (+0.55) |
| White-box CW | 50.61 | 51.60 (+0.99) | 51.53 (+0.92) | 51.77 (+1.16) |
| Black-box PGDtrades | 66.28 | 66.91 (+0.63) | 66.55 (+0.27) | 67.16 (+0.88) |
| Square (query-based) | 80.03 | 80.10 (+0.07) | 80.20 (+0.17) | 80.87 (+0.84) |
| Black-box CW | 64.79 | 65.42 (+0.63) | 65.14 (+0.35) | 65.71 (+0.92) |
| AutoAttack (separate) | 48.88 | 49.36 (+0.48) | 49.25 (+0.37) | 49.35 (+0.47) |
| Seven-attack mean | 61.39 | 61.73 (+0.34) | 61.60 (+0.20) | 62.09 (+0.70) |
| Eight-metric mean | 64.83 | 65.11 (+0.28) | 65.01 (+0.18) | 65.46 (+0.63) |

**G3 exceeds the baseline on six of eight primary metrics:** Clean is lower by 0.11 pp and FGSM by 0.49 pp. **G7 exceeds it on seven of eight:** FGSM remains lower by 0.58 pp. Neither achieves the all-eight-metric target; higher means or AutoAttack do not replace that requirement. G7 is R5's frozen base recipe, while G3 is retained as an earlier aggregate-performance reference. **R5 also exceeds the paper baseline on seven of eight metrics, with the FGSM deficit reduced to 0.12 pp; AA is 49.35% (+0.47 pp).** Its seven-attack/eight-metric means are 62.09%/65.46%. R5 improves all eight primary results over the same-batch R1 control by +0.16 / +0.46 / +0.40 / +0.52 / +0.24 / +0.54 / +0.67 / +0.57 pp, in table order; this does not imply all-eight superiority over the paper.

The [G7 backup README](best_backup/resnet18_cifar10_0906v2/README.md) records its configuration, checkpoint hash, training job 132043 and evaluation job 132061. G7 used target mixing 0.25, non-target mixing 0.00 and reference mixing 0.20; its completed result belongs to G7 alone, not to R5. The backup remains unchanged. The [R5 backup README](best_backup/resnet18_cifar10_0909v1/README.md) records the new source, epoch 230, training/evaluation jobs 132648/132743, original result/log locations and checkpoint SHA256 `865a125f1c82c68e7f3885674b0d72900a1ed7f0182689367f0210d1ed2e021a`; evaluator SHA256 is `26277d6891cd0b6d76050b630a81720cfcae2269caa2d45590bc37c3b76a4164`.

### MobileNet-V2 / CIFAR-10

| Metric | Paper CIARD baseline | 0624 tm010-repeat (Δ vs baseline) |
| --- | ---: | ---: |
| Clean | 89.51 | 89.58 (+0.07) |
| White-box FGSM | 59.10 | 60.12 (+1.02) |
| White-box PGDsat | 47.67 | 49.56 (+1.89) |
| White-box PGDtrades | 50.71 | 52.28 (+1.57) |
| White-box CW | 46.88 | 48.50 (+1.62) |
| Black-box PGDtrades | 66.66 | 67.32 (+0.66) |
| Square (query-based) | 80.01 | 80.78 (+0.77) |
| Black-box CW | 66.12 | 66.18 (+0.06) |
| Seven-attack mean | 59.59 | 60.68 (+1.08) |
| Eight-metric mean | 63.33 | 64.29 (+0.96) |

The completed **0624 tm010-repeat** reference exceeds its paper baseline on all eight primary metrics. This is the base recipe used by the pending M2 candidate, not evidence that M2 improves on it. The [MobileNet backup README](best_backup/mobilenetv2_cifar10/README.md) preserves the configuration and baseline comparison; the local source report is `结果分析/CIARD_Expansion0624_teacher_margin_gate_variants_结果分析.md`. That historical summary does not report AutoAttack, so no MobileNet AutoAttack result is inferred here. In 0909v1, **M1 is the old-recipe control** (one view, AWP=0, consistency=0); its completed eight-metric mean is 64.27%, and its checkpoint SHA256 matches the archived best checkpoint. Those results are not evidence for the active M2 method. M2 uses two views, AWP=0.002 and consistency=0.5 while retaining MobileNet's own base parameters, without ResNet split KD or PCGrad.

### Historical evidence and comparability

Evidence: 0906v1 variant `resnet18_tmix_a020_s120_w40_p081740`, training job
132014, evaluation job 132030, EMA `student_best.pth` checkpoint SHA256
`e18584ba3f84e936f9fbcb80caa4a4297e1df03998841cc256affc22a24d9b89`;
evaluator SHA256 `ee28e4df5279ba8edeca847d5270188db2598279fb66524292533c8c94f75637`.
The completed local result report is `结果分析/0906v1_结果分析.md`; the historical
0703 comparison is recorded in `LOCAL_VERSION_MANAGEMENT.md`. Those local
reports/weights are not bundled here. The preceding
[0906v1 source commit](https://github.com/llmnjust-afk/IJCV_KD/commit/b7c14e7d5ac87f2c66ca9c56aa95deb4dd222c1c)
preserves the evaluated candidate's source. These results belong to 0906v1,
not to G7 or the active 0909v1 R5/M2 entries.

The historical runs use the 50,000-image training protocol and test-loader
checkpoint selection, which introduces selection bias. Stochastic attack seeds
are not all explicitly fixed. PGDtrades uses step size 0.003 in the frozen
evaluator; the paper describes 2/255, so comparison to its published numbers
is not proof of an exactly matched reproduction.

## Other source and execution notes

The previous 0903 MobileNet push=.075 candidate missed the CIARD reference on Clean (89.07 versus 89.51) and black-box CW (65.29 versus 66.12). It has now been replaced in the active entry by M2, based on the verified 0624 push=.05 recipe. The verified reference and all earlier backups remain unchanged. Historical auxiliary MobileNet tools remain available, but the active path uses the copied 4090 wrappers and frozen full evaluator.

Both models retain raw-input WRN-34-10 robust and ResNet-56 natural teachers. Training remains 50,000 images with test-loader checkpoint selection, which introduces selection bias. Stochastic attack seeds are not all explicitly fixed; frozen PGDtrades uses step 0.003 while the paper describes 2/255. Earlier numerical results and future candidate results must retain these qualifications.

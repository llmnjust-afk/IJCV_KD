# IJCV_KD — evaluated CIFAR-10 and CIFAR-100 sources

2026-09-17: updated the user-selected **CIFAR-100 0914v1 ResNet C100-R1 and MobileNet C100-M1** from their completed experiments. Both retain two views, KD-AWP gamma **0.002** and consistency weight **0.50**, and use the original-package WRN-22-6 natural teacher. M1 exceeds the paper on all eight primary metrics; R1 exceeds it on seven attacks, with Clean still 2.20 pp below. CIFAR-10 sources remain evaluated R5/M2.

| Dataset / entry | Source under `run/0914v1` | Previous source backup (old teacher) |
| --- | --- | --- |
| [CIFAR-100 / ResNet-18 C100-R1](CIARD_Expansion_resnet18_cifar100/README.md) | `resnet18_cifar100_natorig_awp0p002_cr0p50` | [0911v1 ResNet](best_backup/resnet18_cifar100_0911v1/README.md) |
| [CIFAR-100 / MobileNet-V2 C100-M1](CIARD_Expansion_mobilenetv2_cifar100/README.md) | `mobilenetv2_cifar100_natorig_awp0p002_cr0p50` | [0911v1 MobileNet](best_backup/mobilenetv2_cifar100_0911v1/README.md) |

The active CIFAR-100 entries contain 45 Python files matching the selected 0914v1 runs byte for byte; requirements are preserved. Wrapper working/log paths, job names and documentation are adapted. The independent 0911v1 backups retain the previous sources and results, with the old natural teacher. Source/result identities, hashes and publication history are in [SYNC_MANIFEST.json](SYNC_MANIFEST.json). These are lightweight source packages; original weights and logs remain in the independent experiments.

## Preserved CIFAR-10 sources

2026-09-11: the CIFAR-10 active entries contain evaluated **ResNet R5** and **MobileNet M2**, both using two adversarial views, KD-AWP gamma **0.002**, consistency weight **0.5**, start120/warmup40 and JS temperature0.5. **R5 exceeds the paper baseline on seven of eight primary metrics, with FGSM 0.12 pp below; M2 also exceeds it on seven, with black-box CW 0.17 pp below.** R5 is preserved in [the 0909v1 backup](best_backup/resnet18_cifar10_0909v1/README.md).

| Entry | Source under `run/0909v1` | Preserved base recipe |
| --- | --- | --- |
| [ResNet-18 R5](CIARD_Expansion_resnet18_cifar10/README.md) | `resnet18_g7_v2_awp0p002_cr0p50` | G7 target=.25/nontarget=0, push=.081740, PCGrad, EMA |
| [MobileNet-V2 M2](CIARD_Expansion_mobilenetv2_cifar10/README.md) | `mobilenetv2_best_v2_awp0p002_cr0p50` | verified push=.05 tm010-repeat, original backpropagation, EMA |

All 40 synchronized CIFAR-10 Python files match their respective experiments byte for byte, including CFG, prefix, model definitions, losses, evaluator and completion checks. No architecture, teacher, data or evaluation protocol change was introduced by synchronization. [SYNC_MANIFEST.json](SYNC_MANIFEST.json) records the source hashes, script adaptations and verification summary. The four existing CIFAR-10 backups remain unchanged; the index also lists the two new CIFAR-100 snapshots.

## Shared method and CIFAR-10 lineage

The first 120 epochs retain the original training path. From epoch 121, each image has two independent original crop/flip views, each with PGD-10 at 8/255 and step 2/255. The full original losses are averaged over views. At epoch 160 the added terms reach their configured strength, with `r=clip((epoch-120)/40,0,1)`.

KD-AWP uses a proxy gradient of the original natural/adversarial distillation objective and applies `v=gamma*r*||w||*g/(||g||+1e-12)` to convolution/linear weights only. The actual full objective is differentiated at perturbed weights; weights are restored before SGD and EMA. Consistency adds `lambda*r*mean(JS(softmax(z_adv1/.5),softmax(z_adv2/.5)))/C`, where C is the number of classes (10 for CIFAR-10 and 100 for CIFAR-100), without a temperature-squared multiplier. Teacher, student and dynamic-temperature updates still occur once per logical batch; ResNet retains its original margin PCGrad.

R5 is the evaluated, user-selected ResNet source: all eight primary results improve over the same-batch G7 control R1. It retains seven improvements over the paper baseline but still misses FGSM by 0.12 pp. R6 has a higher eight-metric mean but three strong white-box results below the paper baseline. M2 applies the R5 combination to MobileNet's verified base recipe. Its completed evaluation improves seven primary metrics over same-batch M1, but black-box CW falls by 0.23 pp; it does not replace M1 as an all-eight-above-paper configuration. Attribution: [Adversarial Weight Perturbation, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/1ef91c212e30e14bf125e9374262401f-Abstract.html) and [Consistency Regularization for Adversarial Robustness, AAAI 2022](https://arxiv.org/pdf/2103.04623). These are CIARD adaptations, not reproductions of the full published recipes or proof of a new contribution. Two views require extra computation.

## CIFAR-10 execution and result status

The existing independent experiments remain the training/evaluation locations. Their source and Slurm scripts are unchanged by this synchronization; no duplicate training is needed. Package wrappers are source templates adapted to the two package directories, with ResNet on compute-4 and MobileNet on compute-2, each requesting one 4090. The package intentionally has no resource symlinks, datasets, checkpoints, outputs or logs, and cannot be submitted directly without preparing an independent run. Both entries use the minimal dependency list for the already verified ciard environment.

Each completed run evaluates one fixed EMA `student_best.pth` on 10,000 test images. R5 training/evaluation jobs 132648/132743 selected epoch 230; M2 jobs 132654/132769 selected epoch 250. All four jobs were verified `COMPLETED / 0:0`. Their original logs remain in the independent experiments; publishing these sources and results requires no repeat training or evaluation. ResNet's target remains all eight metrics strictly above the paper and AA at least 48.88%; higher means alone do not satisfy it. Only the user submits GPU jobs.

## CIFAR-10 results versus the paper CIARD baseline

The tables contain completed **CIFAR-10** evaluations: historical G7 and MobileNet references, plus **0909v1 R5, M1 and M2**. Each configuration uses one selected `student_best.pth`; values are not combined across checkpoints.

Accuracies are percentages. Parentheses show **this result minus the paper CIARD baseline**, in percentage points (pp): `61.30 (-0.58)` means 0.58 pp below the baseline. The baseline is the published CIARD result for the **same student model**, not an earlier version of this repository. Seven-attack and eight-metric means are computed from the corresponding metric values; AutoAttack is separate. Means and differences are computed before rounding, so subtracting two displayed means can differ by 0.01 pp. These means are descriptive summaries, not joint worst-case accuracy or the paper's W-Robust metric.

Baseline sources: [CIARD supplementary material](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Lu_CIARD_Cyclic_Iterative_ICCV_2025_supplemental.pdf), Tables 1/2 for Clean and white-box ResNet/MobileNet results, and Table 5 for AutoAttack; [CIARD paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Lu_CIARD_Cyclic_Iterative_Adversarial_Robustness_Distillation_ICCV_2025_paper.pdf), Tables 5/6, **Robust** columns, for black-box ResNet/MobileNet results. The local copies of these PDFs were used for verification.

### ResNet-18 / CIFAR-10

| Metric | Paper CIARD baseline | 0906v2 G7 (Δ vs baseline) | 0909v1 R5 (Δ vs baseline) |
| --- | ---: | ---: | ---: |
| Clean | 88.87 | 88.91 (+0.04) | 89.07 (+0.20) |
| White-box FGSM | 61.88 | 61.30 (-0.58) | 61.76 (-0.12) |
| White-box PGDsat | 51.70 | 51.91 (+0.21) | 52.34 (+0.64) |
| White-box PGDtrades | 54.46 | 54.54 (+0.08) | 55.01 (+0.55) |
| White-box CW | 50.61 | 51.53 (+0.92) | 51.77 (+1.16) |
| Black-box PGDtrades | 66.28 | 66.55 (+0.27) | 67.16 (+0.88) |
| Square (query-based) | 80.03 | 80.20 (+0.17) | 80.87 (+0.84) |
| Black-box CW | 64.79 | 65.14 (+0.35) | 65.71 (+0.92) |
| AutoAttack (separate) | 48.88 | 49.25 (+0.37) | 49.35 (+0.47) |
| Seven-attack mean | 61.39 | 61.60 (+0.20) | 62.09 (+0.70) |
| Eight-metric mean | 64.83 | 65.01 (+0.18) | 65.46 (+0.63) |

**G7 exceeds the paper baseline on seven of eight primary metrics:** FGSM remains 0.58 pp below. G7 is R5's frozen base recipe. **R5 also exceeds the baseline on seven, with the FGSM deficit reduced to 0.12 pp; AA is 49.35% (+0.47 pp).** Its seven-attack/eight-metric means are 62.09%/65.46%. R5 improves all eight primary results over the same-batch R1 control by +0.16 / +0.46 / +0.40 / +0.52 / +0.24 / +0.54 / +0.67 / +0.57 pp, in table order. Neither G7 nor R5 achieves the all-eight-metric target; higher means or AA do not replace that requirement.

The [G7 backup README](best_backup/resnet18_cifar10_0906v2/README.md) records its configuration, checkpoint hash, training job 132043 and evaluation job 132061. G7 used target mixing 0.25, non-target mixing 0.00 and reference mixing 0.20; its completed result belongs to G7 alone, not to R5. The backup remains unchanged. The [R5 backup README](best_backup/resnet18_cifar10_0909v1/README.md) records the new source, epoch 230, training/evaluation jobs 132648/132743, original result/log locations and checkpoint SHA256 `865a125f1c82c68e7f3885674b0d72900a1ed7f0182689367f0210d1ed2e021a`; evaluator SHA256 is `26277d6891cd0b6d76050b630a81720cfcae2269caa2d45590bc37c3b76a4164`.

### MobileNet-V2 / CIFAR-10

| Metric | Paper CIARD baseline | 0624 tm010-repeat (Δ vs baseline) | 0909v1 M2 (Δ vs baseline) |
| --- | ---: | ---: | ---: |
| Clean | 89.51 | 89.58 (+0.07) | 89.79 (+0.28) |
| White-box FGSM | 59.10 | 60.12 (+1.02) | 61.03 (+1.93) |
| White-box PGDsat | 47.67 | 49.56 (+1.89) | 51.06 (+3.39) |
| White-box PGDtrades | 50.71 | 52.28 (+1.57) | 53.53 (+2.82) |
| White-box CW | 46.88 | 48.50 (+1.62) | 49.14 (+2.26) |
| Black-box PGDtrades | 66.66 | 67.32 (+0.66) | 67.71 (+1.05) |
| Square (query-based) | 80.01 | 80.78 (+0.77) | 81.36 (+1.35) |
| Black-box CW | 66.12 | 66.18 (+0.06) | 65.95 (-0.17) |
| Seven-attack mean | 59.59 | 60.68 (+1.08) | 61.40 (+1.80) |
| Eight-metric mean | 63.33 | 64.29 (+0.96) | 64.95 (+1.61) |

The completed **0624 tm010-repeat** reference and **M1** exceed the paper baseline on all eight primary metrics. M1 is the old-recipe control (one view, AWP=0, consistency=0); its checkpoint SHA256 matches the archived best. The [MobileNet backup README](best_backup/mobilenetv2_cifar10/README.md) preserves that recipe and its historical results. **M2 improves seven primary metrics over M1 and raises the eight-metric mean from 64.27% to 64.95% (+0.68 pp), but black-box CW is 65.95%: 0.23 pp below M1 and 0.17 pp below the paper.** In table order, M2−M1 is +0.21 / +0.91 / +1.61 / +1.31 / +0.64 / +0.38 / +0.58 / -0.23 pp. M2 retains MobileNet's own base parameters, without ResNet split KD or PCGrad; its gains are not an all-metric improvement.

AutoAttack, reported separately: the paper baseline is **46.31%**, M1 is **46.24% (−0.07 pp)** and M2 is **47.15% (+0.84 pp)**; M2 improves over M1 by **0.91 pp**. The historical 0624 summary has no AA result, so none is inferred for that column. The supplementary AA table uses a different MobileNet Clean value; the eight-metric comparison above consistently retains the main Clean baseline of 89.51%.

### Evidence and comparability

M1 training/evaluation jobs are 132653/132747 (epoch 260); M2 jobs are 132654/132769 (epoch 250), all verified `COMPLETED / 0:0`. M2 checkpoint SHA256 is `59566ab81338581b8241db52cb91cceb240e6c437a40fdf84bcf747314629b2a`; evaluator SHA256 is `d3e07b10d6ce533b3c4dc746f8eff7bf13db76250771f4d469fa1bd063e43246`. The M2 experiment retains `logs/train_stdout_132654.log`, `logs/eval_best_stdout_132769.log` and `model/Cifar10_MobileNetV2_0909v1_best_v2_awp0p002_cr0p50/eval_best_0909v1_132769.json`. Source identities and result metadata are recorded in [SYNC_MANIFEST.json](SYNC_MANIFEST.json). The complete local report is `结果分析/0909v1_结果分析.md`; raw logs and weights are not bundled here.

The historical runs use the 50,000-image training protocol and test-loader
checkpoint selection, which introduces selection bias. Stochastic attack seeds
are not all explicitly fixed. PGDtrades uses step size 0.003 in the frozen
evaluator; the paper describes 2/255, so comparison to its published numbers
is not proof of an exactly matched reproduction.

## CIFAR-100 sources, execution and completed results

Both selected entries use 50,000 training / 10,000 test images, 100 classes, 300 epochs, seed 0 and global batch 128. Training uses two 4090 GPUs; evaluation uses one. Default wrappers use compute-4; ResNet also retains a compute-2 training wrapper. Completed R1/M1 training ran on compute-2/compute-4 respectively, and both evaluations ran on compute-2 via a submission override. Wrappers are templates for a future independent run; resources, checkpoints, outputs and completion records are not bundled.

Both keep start120/warmup40 and consistency temperature .5. ResNet retains R5 split KD (target .25, non-target 0, reference .20), push .081740, clean CE .036067, teacher margin .011316 and PCGrad/EMA .999642. MobileNet retains push .05, clean CE .05, teacher margin .01 and EMA .999. These are the new-teacher runs with unchanged base hyperparameters; publication does not modify training, attack implementation or EMA-best selection.

### Natural teacher replacement

Both entries replace the previous WRN-22-6 checkpoint `models/cifar100_wrn_22_6_finetuned_best.pth` (SHA256 `cac8aca0c71e842e958bb49ca7b00fd729731ff465d4f50eb0ec2e53a935cecc`) with the original-package `models/nat_teacher_checkpoint/cifar100_wrn_22_6.pth` (SHA256 `c91c5bf8b5f6c74c427d9a88815c00f98b73a4d1109fb4508b985ae86919e935`). Training configuration and preflight checks bind the same new path/hash; there is no fallback to the old checkpoint. Inputs remain raw [0,1], with CIFAR-100 normalization inside the model and no external Normalize.

Completed teacher diagnostics on RTX 4090 (job133714) and RTX 3090 (job133741) agreed on all 17 cases' per-example class predictions. Under the retained CIFAR-100 processing, old/new natural-teacher Clean was **56.62%/76.49%** on both GPUs. The reason for the old finetuned checkpoint's degradation remains unresolved. This replacement is shared by the selected runs, so its benefit must not be attributed to an additional training-method innovation. Diagnostic evidence paths/hashes are recorded in the manifest; full local report: `结果分析/0914v1_教师诊断_4090与3090对比.md`.

The robust teacher remains WRN-70-16 (`models/cifar100_linf_wrn70-16_without.pt`, SHA256 `3114df6b9d5adf9f275e8fea5a91b71c61df78a568ad31544b6950807d595c8c`), with cyclic updates. Its initial diagnostic Clean of 60.86% versus the paper's 63.56% remains unexplained. CIFAR-10 teachers are unchanged.

Baseline sources: [supplementary material](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Lu_CIARD_Cyclic_Iterative_ICCV_2025_supplemental.pdf), Tables1/2 CIFAR-100 columns for Clean and white-box ResNet/MobileNet; [main paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Lu_CIARD_Cyclic_Iterative_Adversarial_Robustness_Distillation_ICCV_2025_paper.pdf), Tables5/6 CIFAR-100 **Robust** columns for black-box results. Values are percentages; parentheses are result minus the same-model baseline in pp. Means and differences are calculated before rounding. The supplementary AA table is CIFAR-10 and supplies no CIFAR-100 baseline.

### ResNet-18 / CIFAR-100

| Metric | Paper CIARD baseline | 0914v1 C100-R1 (Δ vs baseline) |
| --- | ---: | ---: |
| Clean | 65.73 | 63.53 (-2.20 pp) |
| White-box FGSM | 34.47 | 35.37 (+0.90 pp) |
| White-box PGDsat | 28.05 | 30.91 (+2.86 pp) |
| White-box PGDtrades | 29.45 | 32.02 (+2.57 pp) |
| White-box CW | 24.43 | 27.72 (+3.29 pp) |
| Black-box PGDtrades | 42.29 | 43.35 (+1.06 pp) |
| Square (query-based) | 49.76 | 51.65 (+1.89 pp) |
| Black-box CW | 41.44 | 42.27 (+0.83 pp) |
| Seven-attack mean | 35.70 | 37.61 (+1.91 pp) |
| Eight-metric mean | 39.45 | 40.85 (+1.40 pp) |

**R1 exceeds the paper on all seven attacks; Clean is 63.53%, still 2.20 pp below the paper.** Relative to the old-teacher 0911v1 R1, Clean rises by 0.53 pp, while FGSM and white-box PGDtrades decrease by 0.06 and 0.12 pp. This is the user-selected source, without a claim of improvement on every metric.

AutoAttack, separate absolute measurement: **25.70%**, with no CIFAR-100 paper AA baseline. Training/evaluation jobs **133834/134513** were verified `COMPLETED / 0:0`; fixed EMA best epoch **200**. Configuration, checkpoint/evaluator hashes and original evidence locations are in the [active entry](CIARD_Expansion_resnet18_cifar100/README.md). The [0911v1 backup](best_backup/resnet18_cifar100_0911v1/README.md) preserves the previous source and its old-teacher results.

### MobileNet-V2 / CIFAR-100

| Metric | Paper CIARD baseline | 0914v1 C100-M1 (Δ vs baseline) |
| --- | ---: | ---: |
| Clean | 66.72 | 66.91 (+0.19 pp) |
| White-box FGSM | 33.56 | 34.57 (+1.01 pp) |
| White-box PGDsat | 27.02 | 28.61 (+1.59 pp) |
| White-box PGDtrades | 28.95 | 30.06 (+1.11 pp) |
| White-box CW | 25.54 | 27.28 (+1.74 pp) |
| Black-box PGDtrades | 42.70 | 44.82 (+2.12 pp) |
| Square (query-based) | 50.85 | 53.29 (+2.44 pp) |
| Black-box CW | 42.85 | 43.67 (+0.82 pp) |
| Seven-attack mean | 35.92 | 37.47 (+1.55 pp) |
| Eight-metric mean | 39.77 | 41.15 (+1.38 pp) |

**M1 exceeds the paper on all eight primary metrics from one fixed checkpoint, with Clean 66.91% (+0.19 pp).** All eight also improve over the old-teacher 0911v1 M1. The small Clean margin and single training seed do not establish statistical significance or stability across runs.

AutoAttack, separate absolute measurement: **24.95%**, with no CIFAR-100 paper AA baseline. Training/evaluation jobs **133792/134582** were verified `COMPLETED / 0:0`; fixed EMA best epoch **230**. Configuration, checkpoint/evaluator hashes and original evidence locations are in the [active entry](CIARD_Expansion_mobilenetv2_cifar100/README.md). The [0911v1 backup](best_backup/mobilenetv2_cifar100_0911v1/README.md) preserves the previous source and its old-teacher results.

### CIFAR-100 evidence and interpretation

The complete local report is `结果分析/0914v1_cifar100_结果分析.md`; this publication selects R1 and M1 only. Original source/result/log/checkpoint hashes, 100-class CPU strict loading, training completion, Slurm success and full 10,000-image log/JSON/count agreement were verified. Every row comes from one fixed EMA checkpoint. The result filenames retain `eval_best_0909v1_<job>.json` for historical compatibility; their actual variants belong to 0914v1. Original 0911v1 entries and verification metadata remain in the manifest history and independent backups.

CIFAR-100 retains test-loader `(Clean+PGD proxy)/2` selection of EMA best, introducing selection bias. Stochastic attacks are not all explicitly seeded. The retained evaluation uses L-infinity8/255, PGDsat20 steps/2/255, PGDtrades20 steps/.003, CW30 steps/2/255 and Square100queries. Black-box PGDtrades/CW transfer from WRN-70-16; Square queries the student. The official PGDtrades evaluator's .003 differs from the paper text's 2/255; training remains PGD-10 at 2/255. These are single-seed comparisons to published values, not proof of exactly matched reproduction or statistical stability. Mean accuracy is not joint worst-case accuracy. Only the user submits GPU jobs.

## Other CIFAR-10 source and execution notes

The previous 0903 MobileNet push=.075 candidate missed the CIARD reference on Clean (89.07 versus 89.51) and black-box CW (65.29 versus 66.12). It has now been replaced in the active entry by M2, based on the verified 0624 push=.05 recipe. The verified reference and all earlier backups remain unchanged. Historical auxiliary MobileNet tools remain available, but the active path uses the copied 4090 wrappers and frozen full evaluator.

Both CIFAR-10 models retain raw-input WRN-34-10 robust and ResNet-56 natural teachers. Training remains 50,000 images with test-loader checkpoint selection, which introduces selection bias. Stochastic attack seeds are not all explicitly fixed; frozen PGDtrades uses step 0.003 while the paper describes 2/255. Earlier numerical results and future candidate results must retain these qualifications.

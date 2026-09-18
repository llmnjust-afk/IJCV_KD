# IJCV_KD — evaluated CIFAR-10 and CIFAR-100 sources

2026-09-19: synchronized evaluated sources and results, with publication authorized by the user. CIFAR-10 now uses **0917v1 MobileNet M1 and ResNet R4**. Both **M1 and M4** completed all eight primary paper thresholds and the separate AA threshold at their own checkpoints; M1 is the default source and M4 has [complete reproduction instructions](CIARD_Expansion_mobilenetv2_cifar10/M4_REPRODUCTION.md). R4 remains 0.08 pp below the paper on FGSM. CIFAR-100 retains **0914v1 ResNet R2 only**, using the canonical directory without an _r2 suffix, and MobileNet M1. These are user-selected sources, not a claim of dominance over every historical result.

| Dataset / active entry | Evaluated source | AWP gamma / consistency | Paper primary metrics | AA |
| --- | --- | --- | --- | ---: |
| [CIFAR-10 MobileNet M1](CIARD_Expansion_mobilenetv2_cifar10/README.md) | run/0917v1/mobilenetv2_cifar10_v2_awp0p001_cr0p50 | .001 / .50 | 8/8 | 47.01 |
| [CIFAR-10 ResNet R4](CIARD_Expansion_resnet18_cifar10/README.md) | run/0917v1/resnet18_cifar10_v2_awp0p002_cr0p75 | .002 / .75 | 7/8; FGSM −.08 pp | 49.10 |
| [CIFAR-100 ResNet R2](CIARD_Expansion_resnet18_cifar100/README.md) | run/0914v1/resnet18_cifar100_natorig_awp0p003_cr0p50 | .003 / .50 | 7/8; Clean −1.95 pp | 25.59 |
| [CIFAR-100 MobileNet M1](CIARD_Expansion_mobilenetv2_cifar100/README.md) | run/0914v1/mobilenetv2_cifar100_natorig_awp0p002_cr0p50 | .002 / .50 | 8/8 | 24.95 |

All 88 active Python files match their evaluated runs byte for byte. Three new [dated source backups](best_backup/README.md) contain CIFAR-10 M1, CIFAR-10 R4 and CIFAR-100 M1; their 67 Python files also match the evaluated sources. Backup names use the experiment batch date (0917v1 or 0914v1); synchronization date is 2026-09-19. All six earlier backup directories remain unchanged and are historical references. Full configurations, source/result/checkpoint hashes and selection history are in [SYNC_MANIFEST.json](SYNC_MANIFEST.json). No separate M4 source tree is needed: only two CFG values and four identity/path assignments differ from M1.

## Shared method and CIFAR-10 lineage

The first 120 epochs retain the original training path. From epoch 121, each image has two independent original crop/flip views, each with PGD-10 at 8/255 and step 2/255. The full original losses are averaged over views. At epoch 160 the added terms reach their configured strength, with `r=clip((epoch-120)/40,0,1)`.

KD-AWP uses a proxy gradient of the original natural/adversarial distillation objective and applies `v=gamma*r*||w||*g/(||g||+1e-12)` to convolution/linear weights only. The actual full objective is differentiated at perturbed weights; weights are restored before SGD and EMA. Consistency adds `lambda*r*mean(JS(softmax(z_adv1/.5),softmax(z_adv2/.5)))/C`, where C is the number of classes (10 for CIFAR-10 and 100 for CIFAR-100), without a temperature-squared multiplier. Teacher, student and dynamic-temperature updates still occur once per logical batch; ResNet retains its original margin PCGrad.

CIFAR-10 M1/M4 and R4 inherit the evaluated 0909v1 method from the 0914v1 source package. MobileNet retains its own push=.05, ordinary backpropagation and EMA=.999; ResNet retains split target/non-target mixing .25/0, push=.081740, PCGrad and EMA=.999642. Only the stated AWP/consistency values and experiment identities differ from those prior entries. Attribution: [Adversarial Weight Perturbation, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/1ef91c212e30e14bf125e9374262401f-Abstract.html) and [Consistency Regularization for Adversarial Robustness, AAAI 2022](https://arxiv.org/pdf/2103.04623). These are CIARD adaptations, not reproductions of the full published recipes or proof of a new contribution.

## CIFAR-10 completed results

Each column uses one fixed EMA student_best.pth evaluated on all 10,000 test images; individual best metrics are not combined across checkpoints. Values are percentages, and parentheses are result minus the same-model paper baseline in percentage points. Seven-attack/eight-metric means exclude AA and are calculated before rounding; they are not joint worst-case accuracy or W-Robust.

Baseline sources, verified using local PDFs: [CIARD supplementary material](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Lu_CIARD_Cyclic_Iterative_ICCV_2025_supplemental.pdf), Tables1/2 for Clean/white-box and Table5 for CIFAR-10 AA; [CIARD paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Lu_CIARD_Cyclic_Iterative_Adversarial_Robustness_Distillation_ICCV_2025_paper.pdf), Tables5/6 **Robust** columns for black-box results. MobileNet Clean consistently uses89.51%, not the different Clean value in the AA table.

### MobileNet-V2 / CIFAR-10

| Metric | Paper CIARD baseline | 0909v1 M2 (historical) (Δ vs paper) | 0917v1 M1 (Δ vs paper) | 0917v1 M4 (Δ vs paper) |
| --- | ---: | ---: | ---: | ---: |
| Clean | 89.51 | 89.79 (+0.28 pp) | 89.61 (+0.10 pp) | 90.02 (+0.51 pp) |
| White-box FGSM | 59.10 | 61.03 (+1.93 pp) | 60.87 (+1.77 pp) | 60.71 (+1.61 pp) |
| White-box PGDsat | 47.67 | 51.06 (+3.39 pp) | 50.67 (+3.00 pp) | 50.75 (+3.08 pp) |
| White-box PGDtrades | 50.71 | 53.53 (+2.82 pp) | 53.36 (+2.65 pp) | 53.34 (+2.63 pp) |
| White-box CW | 46.88 | 49.14 (+2.26 pp) | 49.01 (+2.13 pp) | 48.76 (+1.88 pp) |
| Black-box PGDtrades | 66.66 | 67.71 (+1.05 pp) | 67.67 (+1.01 pp) | 67.98 (+1.32 pp) |
| Square (query-based) | 80.01 | 81.36 (+1.35 pp) | 81.27 (+1.26 pp) | 81.41 (+1.40 pp) |
| Black-box CW | 66.12 | 65.95 (-0.17 pp) | 66.14 (+0.02 pp) | 66.14 (+0.02 pp) |
| AutoAttack (separate) | 46.31 | 47.15 (+0.84 pp) | 47.01 (+0.70 pp) | 46.70 (+0.39 pp) |
| Seven-attack mean | 59.59 | 61.40 (+1.80 pp) | 61.28 (+1.69 pp) | 61.30 (+1.71 pp) |
| Eight-metric mean | 63.33 | 64.95 (+1.61 pp) | 64.83 (+1.49 pp) | 64.89 (+1.56 pp) |

M1 is the user-selected default: FGSM, white-box PGDtrades/CW and AA are higher than M4. M4 has higher Clean, PGDsat, black-box PGDtrades and Square, and a higher eight-metric mean (64.89% versus64.83%; M1 exact64.825%). Black-box CW ties at66.14%, only0.02pp above the paper. M4−M1 in the nine-metric order is +.41/−.16/+.08/−.02/−.25/+.31/+.14/0/−.31pp. Both pass the paper thresholds in this single-seed evaluation, but neither dominates historical0909v1 M2. [M4 reproduction instructions](CIARD_Expansion_mobilenetv2_cifar10/M4_REPRODUCTION.md) are also included inside the M1 backup.

### ResNet-18 / CIFAR-10

| Metric | Paper CIARD baseline | 0909v1 R5 (historical) (Δ vs paper) | 0917v1 R4 (Δ vs paper) |
| --- | ---: | ---: | ---: |
| Clean | 88.87 | 89.07 (+0.20 pp) | 89.10 (+0.23 pp) |
| White-box FGSM | 61.88 | 61.76 (-0.12 pp) | 61.80 (-0.08 pp) |
| White-box PGDsat | 51.70 | 52.34 (+0.64 pp) | 52.06 (+0.36 pp) |
| White-box PGDtrades | 54.46 | 55.01 (+0.55 pp) | 54.86 (+0.40 pp) |
| White-box CW | 50.61 | 51.77 (+1.16 pp) | 51.24 (+0.63 pp) |
| Black-box PGDtrades | 66.28 | 67.16 (+0.88 pp) | 67.28 (+1.00 pp) |
| Square (query-based) | 80.03 | 80.87 (+0.84 pp) | 80.77 (+0.74 pp) |
| Black-box CW | 64.79 | 65.71 (+0.92 pp) | 65.62 (+0.83 pp) |
| AutoAttack (separate) | 48.88 | 49.35 (+0.47 pp) | 49.10 (+0.22 pp) |
| Seven-attack mean | 61.39 | 62.09 (+0.70 pp) | 61.95 (+0.55 pp) |
| Eight-metric mean | 64.83 | 65.46 (+0.63 pp) | 65.34 (+0.51 pp) |

R4 is retained by user choice. It exceeds seven primary paper metrics, with FGSM61.80% still0.08pp below61.88%; a strict gain would require at least61.89%. AA49.10% is0.22pp above the paper. Compared with R5, FGSM rises0.04pp while the mean and AA decrease. R4 does not achieve all-eight improvement and is not a general replacement for R5's numerical advantages. The [0909v1 R5 backup](best_backup/resnet18_cifar10_0909v1/README.md) and [0906v2 G7 backup](best_backup/resnet18_cifar10_0906v2/README.md) retain their original evidence. The [0624 MobileNet reference](best_backup/mobilenetv2_cifar10/README.md) also remains historical; its results must not be confused with0917v1 M1.

### CIFAR-10 evidence and execution

| Configuration | EMA best epoch | Training / evaluation jobs | Actual node / GPUs |
| --- | ---: | --- | --- |
| 0917v1 M1 | 250 | 134636 / 135053 | compute-2 / 1×4090 |
| 0917v1 M4 | 254 | 134639 / 135056 | compute-2 / 1×4090 |
| 0917v1 R4 | 252 | 134643 / 134905 | compute-4 / 1×4090 |

All six jobs are COMPLETED / 0:0. CPU strict checkpoint loading, source/evaluator/checkpoint hashes, complete nine-metric correct counts, log/JSON agreement and Slurm stdout/stderr checks passed. Per-run paths and hashes are in the source READMEs, M4 guide and manifest; full local report: 结果分析/0917v1_结果分析.md. Original sources, outputs and weights remain in run/0917v1.

Source-package Slurm templates only adapt working and log paths. They retain the tested4090 resources. Packages contain no data/models links, checkpoints, model/logs or caches and cannot be submitted directly. Future reproduction requires a new independent run with unique identity/prefix, resource links, output directories and matching script paths. Only the user submits GPU jobs; this local synchronization requires no repeat training or evaluation.

All runs retain50,000-image training and test-loader (Clean+PGD proxy)/2 checkpoint selection, introducing selection bias. Stochastic attacks are not all explicitly seeded. Frozen PGDtrades uses step.003 while the paper describes2/255; training remains PGD-10 at2/255. These are comparisons of single-seed results to published values, not proof of identical reproduction or stable multi-seed superiority.

## CIFAR-100 sources, execution and completed results

All four experiments use 50,000 training / 10,000 test images, 100 classes, 300 epochs, seed 0, global batch 128 and two adversarial views. Training uses two 4090 GPUs; evaluation uses one. Both ResNet runs trained on compute-2 and both MobileNet runs on compute-4. R1/M1/M2 evaluations ran on compute-2; R2 evaluation ran on compute-4. Default package wrappers use compute-4, and the retained R2 entry includes a compute-2 training alternative. Wrappers are templates for future independent runs, not ready-to-submit experiments with bundled weights or completion records.

| Configuration | AWP gamma | Consistency weight | EMA best epoch | Train / evaluation jobs | Evaluation node | Status |
| --- | ---: | ---: | ---: | --- | --- | --- |
| C100-R1 | 0.002 | 0.50 | 200 | 133834 / 134513 | compute-2 | COMPLETED / 0:0 |
| C100-R2 | 0.003 | 0.50 | 200 | 134287 / 134829 | compute-4 | COMPLETED / 0:0 |
| C100-M1 | 0.002 | 0.50 | 230 | 133792 / 134582 | compute-2 | COMPLETED / 0:0 |
| C100-M2 | 0.002 | 0.75 | 230 | 133793 / 134633 | compute-2 | COMPLETED / 0:0 |

All retain start120/warmup40 and consistency temperature .5. ResNet retains split KD (target .25, non-target 0, reference .20), push .081740, clean CE .036067, teacher margin .011316 and PCGrad/EMA .999642. MobileNet retains push .05, clean CE .05, teacher margin .01 and EMA .999. R1/M1 use the new teacher with the previous base hyperparameters; R2 changes only AWP gamma relative to R1, and M2 only consistency weight relative to M1. Publication does not change training, attack implementation or fixed EMA-best selection.

### Natural teacher replacement

All four experiments replace the previous WRN-22-6 checkpoint `models/cifar100_wrn_22_6_finetuned_best.pth` (SHA256 `cac8aca0c71e842e958bb49ca7b00fd729731ff465d4f50eb0ec2e53a935cecc`) with the original-package `models/nat_teacher_checkpoint/cifar100_wrn_22_6.pth` (SHA256 `c91c5bf8b5f6c74c427d9a88815c00f98b73a4d1109fb4508b985ae86919e935`). Training configuration and preflight checks bind the same new path/hash; there is no fallback to the old checkpoint. Inputs remain raw [0,1], with CIFAR-100 normalization inside the model and no external Normalize.

Completed teacher diagnostics on RTX 4090 (job133714) and RTX 3090 (job133741) agreed on all 17 cases' per-example class predictions. Under the retained CIFAR-100 processing, old/new natural-teacher Clean was **56.62%/76.49%** on both GPUs. The reason for the old finetuned checkpoint's degradation remains unresolved. This replacement is shared by all four runs, so its benefit must not be attributed to an additional training-method innovation. Diagnostic evidence paths/hashes are recorded in the manifest; full local report: `结果分析/0914v1_教师诊断_4090与3090对比.md`.

The robust teacher remains WRN-70-16 (`models/cifar100_linf_wrn70-16_without.pt`, SHA256 `3114df6b9d5adf9f275e8fea5a91b71c61df78a568ad31544b6950807d595c8c`), with cyclic updates. Its initial diagnostic Clean of 60.86% versus the paper's 63.56% remains unexplained. CIFAR-10 teachers are unchanged.

Baseline sources: [supplementary material](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Lu_CIARD_Cyclic_Iterative_ICCV_2025_supplemental.pdf), Tables1/2 CIFAR-100 columns for Clean and white-box ResNet/MobileNet; [main paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Lu_CIARD_Cyclic_Iterative_Adversarial_Robustness_Distillation_ICCV_2025_paper.pdf), Tables5/6 CIFAR-100 **Robust** columns for black-box results. Values are percentages; parentheses are result minus the same-model baseline in pp. Means and differences are calculated before rounding. The supplementary AA table is CIFAR-10 and supplies no CIFAR-100 baseline.

### ResNet-18 / CIFAR-100

| Metric | Paper CIARD baseline | 0914v1 C100-R1 (Δ vs baseline) | 0914v1 C100-R2 (Δ vs baseline) |
| --- | ---: | ---: | ---: |
| Clean | 65.73 | 63.53 (-2.20 pp) | 63.78 (-1.95 pp) |
| White-box FGSM | 34.47 | 35.37 (+0.90 pp) | 35.70 (+1.23 pp) |
| White-box PGDsat | 28.05 | 30.91 (+2.86 pp) | 31.20 (+3.15 pp) |
| White-box PGDtrades | 29.45 | 32.02 (+2.57 pp) | 32.28 (+2.83 pp) |
| White-box CW | 24.43 | 27.72 (+3.29 pp) | 27.82 (+3.39 pp) |
| Black-box PGDtrades | 42.29 | 43.35 (+1.06 pp) | 43.29 (+1.00 pp) |
| Square (query-based) | 49.76 | 51.65 (+1.89 pp) | 51.69 (+1.93 pp) |
| Black-box CW | 41.44 | 42.27 (+0.83 pp) | 42.49 (+1.05 pp) |
| Seven-attack mean | 35.70 | 37.61 (+1.91 pp) | 37.78 (+2.08 pp) |
| Eight-metric mean | 39.45 | 40.85 (+1.40 pp) | 41.03 (+1.58 pp) |

**Only R2 is retained as an active source in this package, under CIARD_Expansion_resnet18_cifar100. R1 is a historical result reference; its source remains in origin_code/0914v1 and its original run.** R2−R1 in table order is +0.25 / +0.33 / +0.29 / +0.26 / +0.10 / −0.06 / +0.04 / +0.22 pp: seven primary metrics rise and black-box PGDtrades falls. AutoAttack is **25.70% for R1 and 25.59% for R2**. The higher R2 eight-metric mean does not imply improvement on every metric; neither run exceeds the paper's Clean (R1/R2 gaps: 2.20/1.95 pp). These are single-seed observations, not statistically established superiority. The R2 source README binds its configuration, checkpoint and evaluation evidence; historical R1 evidence remains in the manifest.

### MobileNet-V2 / CIFAR-100

| Metric | Paper CIARD baseline | 0914v1 C100-M1 (Δ vs baseline) | 0914v1 C100-M2 (Δ vs baseline) |
| --- | ---: | ---: | ---: |
| Clean | 66.72 | 66.91 (+0.19 pp) | 66.88 (+0.16 pp) |
| White-box FGSM | 33.56 | 34.57 (+1.01 pp) | 34.47 (+0.91 pp) |
| White-box PGDsat | 27.02 | 28.61 (+1.59 pp) | 28.59 (+1.57 pp) |
| White-box PGDtrades | 28.95 | 30.06 (+1.11 pp) | 30.09 (+1.14 pp) |
| White-box CW | 25.54 | 27.28 (+1.74 pp) | 27.13 (+1.59 pp) |
| Black-box PGDtrades | 42.70 | 44.82 (+2.12 pp) | 44.56 (+1.86 pp) |
| Square (query-based) | 50.85 | 53.29 (+2.44 pp) | 53.55 (+2.70 pp) |
| Black-box CW | 42.85 | 43.67 (+0.82 pp) | 43.73 (+0.88 pp) |
| Seven-attack mean | 35.92 | 37.47 (+1.55 pp) | 37.45 (+1.52 pp) |
| Eight-metric mean | 39.77 | 41.15 (+1.38 pp) | 41.12 (+1.35 pp) |

**Both MobileNet runs exceed all eight primary paper values at their own fixed checkpoints.** Clean margins are only 0.19/0.16 pp. M2−M1 is −0.03 / −0.10 / −0.02 / +0.03 / −0.15 / −0.26 / +0.26 / +0.06 pp, with three increases and five decreases. M1 remains the active source; M2 retains advantages in white-box PGDtrades, Square and black-box CW. AutoAttack is **24.95% for M1 and 24.53% for M2**. M2's exact eight-metric mean is 41.125%, displayed as 41.12% following the original log's rounding. The small differences do not establish multi-seed stability or statistical significance.

### CIFAR-100 evidence and interpretation

The complete local report is `结果分析/0914v1_cifar100_结果分析.md`; all four configurations are now reported. All eight training/evaluation jobs completed successfully. Source/result/log/checkpoint hashes, 100-class CPU strict loading and full 10,000-image log/JSON/count agreement were verified. Each row uses one fixed EMA checkpoint, with no selection of best individual metrics across checkpoints. This package retains R2 and M1 source entries only; R1 and M2 source remain in their original independent runs, with all four full configurations and source hashes in the manifest. Prior R1/R2/M1 publication belongs to origin_code/0914v1, which is unchanged. Result filenames retain `eval_best_0909v1_<job>.json` for historical compatibility; actual variants belong to 0914v1. Previous publication/verification records and frozen-source mappings remain available.

CIFAR-100 retains test-loader `(Clean+PGD proxy)/2` selection of EMA best, introducing selection bias. Stochastic attacks are not all explicitly seeded. The retained evaluation uses L-infinity8/255, PGDsat20 steps/2/255, PGDtrades20 steps/.003, CW30 steps/2/255 and Square100queries. Black-box PGDtrades/CW transfer from WRN-70-16; Square queries the student. The official PGDtrades evaluator's .003 differs from the paper text's 2/255; training remains PGD-10 at 2/255. These are single-seed comparisons to published values, not proof of exactly matched reproduction or statistical stability. Mean accuracy is not joint worst-case accuracy. Only the user submits GPU jobs.

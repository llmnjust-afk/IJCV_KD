# IJCV_KD — evaluated CIFAR-10 and CIFAR-100 sources

2026-10-05 **H2 source update; GitHub publication authorized by the user**: the user-selected CIFAR-100 ResNet entry is now **1004 H2**, replacing R2 under the canonical `CIARD_Expansion_resnet18_cifar100` directory. H2 exceeds all eight primary paper values in its single-seed, fixed epoch-190 EMA evaluation under the project's frozen protocol. The other three active entries remain unchanged. The user authorized publication of this verified source update to `llmnjust-afk/IJCV_KD` on `main`; no training or evaluation jobs are submitted.

**Historical publication record — 2026-09-19:** evaluated sources and results were synchronized with publication authorized by the user. CIFAR-10 selected **0917v1 MobileNet M1 and ResNet R4**. Both **M1 and M4** completed all eight primary paper thresholds and the separate AA threshold at their own checkpoints; M1 remains the default source and M4 has [complete reproduction instructions](CIARD_Expansion_mobilenetv2_cifar10/M4_REPRODUCTION.md). R4 remains 0.08 pp below the paper on FGSM. At that time CIFAR-100 retained **0914v1 ResNet R2** and MobileNet M1. The R2 entry is superseded locally by H2 on 2026-10-05; the historical record and results remain intact. These are user-selected sources, not a claim of dominance over every historical result.

| Dataset / active entry | Evaluated source | AWP gamma / consistency | Paper primary metrics | AA |
| --- | --- | --- | --- | ---: |
| [CIFAR-10 MobileNet M1](CIARD_Expansion_mobilenetv2_cifar10/README.md) | run/0917v1/mobilenetv2_cifar10_v2_awp0p001_cr0p50 | .001 / .50 | 8/8 | 47.01 |
| [CIFAR-10 ResNet R4](CIARD_Expansion_resnet18_cifar10/README.md) | run/0917v1/resnet18_cifar10_v2_awp0p002_cr0p75 | .002 / .75 | 7/8; FGSM −.08 pp | 49.10 |
| [CIFAR-100 ResNet H2](CIARD_Expansion_resnet18_cifar100/README.md) | run/1004-cifar100-binary-v1/resnet18_cifar100_nb_l2_w40_s0 | .003 / .50 | 8/8 this seed; minimum FGSM +.05 pp | 24.22 |
| [CIFAR-100 MobileNet M1](CIARD_Expansion_mobilenetv2_cifar100/README.md) | run/0914v1/mobilenetv2_cifar100_natorig_awp0p002_cr0p50 | .002 / .50 | 8/8 | 24.95 |

The package now contains **117 active Python files**, replacing the prior total of 88. H2 contributes 50 files, including its 25-file `frozen_source` snapshot; the unchanged other three entries contribute 67. Active Python files match their evaluated runs byte for byte. A new [H2 source backup](best_backup/resnet18_cifar100_1004_h2/README.md) preserves the same 50 Python files. The three backups added on 2026-09-19 retain their 67 files, so these four dated additions total 117 Python files. **All nine pre-existing backup directories remain frozen.** Backup names use experiment batch dates; this H2 synchronization occurred on 2026-10-05. Full configurations, source/result/checkpoint hashes and historical mappings are in [SYNC_MANIFEST.json](SYNC_MANIFEST.json). No separate M4 source tree is needed: only two CFG values and four identity/path assignments differ from M1.

This is a **source snapshot, not a ready-to-submit experiment**: data/model resources, output directories, checkpoints and logs are not bundled. H2's Python files deliberately retain the evaluated variant/prefix, absolute checkpoint path, `run_checks.py` identity checks and parent-batch manifest bindings. Its A800 scripts only adapt package working/log paths and job names; moving these scripts does not make those frozen bindings valid here. A new standalone run requires a new independent identity, matching manifest/hash bindings, resources and output paths. Do not submit the snapshot scripts or reuse the completed original H2 prefix. Only the user submits GPU jobs.

## Shared method and CIFAR-10 lineage

The first 120 epochs retain the original training path. From epoch 121, each image has two independent original crop/flip views, each with PGD-10 at 8/255 and step 2/255. Losses are averaged over views. AWP and consistency reach their configured strength at epoch 160, with `r=clip((epoch-120)/40,0,1)`; the separate CE and teacher-margin schedules remain source-specific.

KD-AWP uses a proxy gradient of each entry's natural/adversarial distillation objective and applies `v=gamma*r*||w||*g/(||g||+1e-12)` to convolution/linear weights only. ResNet's adversarial KD includes its retained target-mix/split correction; MobileNet leaves that correction disabled. The actual full objective is differentiated at perturbed weights; weights are restored before SGD and EMA. Consistency adds `lambda*r*mean(JS(softmax(z_adv1/.5),softmax(z_adv2/.5)))/C`, where C is the number of classes (10 for CIFAR-10 and 100 for CIFAR-100), without a temperature-squared multiplier. Teacher, student and dynamic-temperature updates still occur once per logical batch; ResNet retains its original margin PCGrad.

CIFAR-10 M1/M4 and R4 inherit the evaluated 0909v1 method from the 0914v1 source package. MobileNet retains its own push=.05, ordinary backpropagation and EMA=.999; ResNet retains split target/non-target mixing .25/0, push=.081740, PCGrad and EMA=.999642. Only the stated AWP/consistency values and experiment identities differ from those prior entries. Attribution: [Adversarial Weight Perturbation, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/1ef91c212e30e14bf125e9374262401f-Abstract.html) and [Consistency Regularization for Adversarial Robustness, AAAI 2022](https://arxiv.org/pdf/2103.04623). These are CIARD adaptations, not reproductions of the full published recipes or proof of a new contribution.

**H2 adds `natural_binary_outer` on top of its retained ResNet recipe:** weight2, start120/warmup40, first active at epoch121 and full strength at160. The extra true-versus-rest natural-teacher KL applies only when the natural teacher is correct, the student's clean prediction is wrong and the teacher's true-class probability is higher. Teacher targets and masks are detached; the teacher uses the current natural-KD temperature, the student uses temperature1, and the loss is averaged over the full batch then divided by100 classes, without a T-squared multiplier. It is added only to the student outer objective; the original natural KD remains, and the extra term does not enter the AWP inner objective or the original natural-KD dynamics metric. H2 retains reference target mix .20, split target/non-target .25/0 and teacher-margin PCGrad; ELS, CE gate transfer and CE projection are disabled. Thus the entries share a CIARD extension framework, but the existing MobileNet result is not a cross-architecture validation of H2's complete active method or its new binary term.

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

**Historical 0914v1 R1/R2/M1/M2:** all four experiments used 50,000 training / 10,000 test images, 100 classes, 300 epochs, seed0, global batch128 and two adversarial views after epoch120. Training used two4090 GPUs; evaluation used one. Both ResNet runs trained on compute-2 and both MobileNet runs on compute-4. R1/M1/M2 evaluations ran on compute-2; R2 evaluation ran on compute-4. Their wrapper/resource descriptions belong to those historical runs; the former R2 compute-2 alternative is not an H2 execution script.

**Current H2:** from-scratch training uses all50,000 training images, seed0, batch128 and one A800, retaining the original300-epoch learning-rate formula but ending at190. The only formal checkpoint is the EMA after the final epoch190 update, `student_epoch190.pth`; training does not load the test set or select a checkpoint from test accuracy. The method and endpoint were informed by historical test results, so the project test set is not an untouched research holdout. Training137534 and full evaluation137729 both completed on compute-6 with one A800. The package preserves source bindings, not runnable weights/completion records.

| Configuration | AWP gamma | Consistency weight | EMA checkpoint epoch | Train / evaluation jobs | Evaluation node | Status |
| --- | ---: | ---: | ---: | --- | --- | --- |
| C100-R1 | 0.002 | 0.50 | 200 | 133834 / 134513 | compute-2 | COMPLETED / 0:0 |
| C100-R2 | 0.003 | 0.50 | 200 | 134287 / 134829 | compute-4 | COMPLETED / 0:0 |
| C100-M1 | 0.002 | 0.50 | 230 | 133792 / 134582 | compute-2 | COMPLETED / 0:0 |
| C100-M2 | 0.002 | 0.75 | 230 | 133793 / 134633 | compute-2 | COMPLETED / 0:0 |
| 1004 H2 | 0.003 | 0.50 | 190 (fixed final EMA) | 137534 / 137729 | compute-6 | COMPLETED / 0:0 |

The historical 0914 runs retain method start120/warmup40 and consistency temperature .5. ResNet retains split KD (target .25, non-target 0, reference .20), push .081740, clean CE .036067, teacher margin .011316 and PCGrad/EMA .999642. MobileNet retains push .05, clean CE .05, teacher margin .01 and EMA .999. R1/M1 use the new teacher with the previous base hyperparameters; R2 changes only AWP gamma relative to R1, and M2 only consistency weight relative to M1. H2 preserves that ResNet component family with the added outer binary term described above; it uses fixed final EMA selection instead of the historical EMA-best rule. This local synchronization changes neither evaluated training nor attack implementations.

### Natural teacher replacement

The four historical 0914 experiments replaced the previous WRN-22-6 checkpoint `models/cifar100_wrn_22_6_finetuned_best.pth` (SHA256 `cac8aca0c71e842e958bb49ca7b00fd729731ff465d4f50eb0ec2e53a935cecc`) with the original-package `models/nat_teacher_checkpoint/cifar100_wrn_22_6.pth` (SHA256 `c91c5bf8b5f6c74c427d9a88815c00f98b73a4d1109fb4508b985ae86919e935`). H2 retains this original-package teacher. Training configuration and preflight checks bind the same path/hash; there is no fallback to the old checkpoint. Inputs remain raw [0,1], with CIFAR-100 normalization inside the model and no external Normalize.

Completed teacher diagnostics on RTX 4090 (job133714) and RTX 3090 (job133741) agreed on all 17 cases' per-example class predictions. Under the retained CIFAR-100 processing, old/new natural-teacher Clean was **56.62%/76.49%** on both GPUs. The reason for the old finetuned checkpoint's degradation remains unresolved. This replacement is shared by all four runs, so its benefit must not be attributed to an additional training-method innovation. Diagnostic evidence paths/hashes are recorded in the manifest; full local report: `结果分析/0914v1_教师诊断_4090与3090对比.md`.

The robust teacher remains WRN-70-16 (`models/cifar100_linf_wrn70-16_without.pt`, SHA256 `3114df6b9d5adf9f275e8fea5a91b71c61df78a568ad31544b6950807d595c8c`), with cyclic updates. Its initial diagnostic Clean of 60.86% versus the paper's 63.56% remains unexplained. CIFAR-10 teachers are unchanged.

Baseline sources: [supplementary material](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Lu_CIARD_Cyclic_Iterative_ICCV_2025_supplemental.pdf), Tables1/2 CIFAR-100 columns for Clean and white-box ResNet/MobileNet; [main paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Lu_CIARD_Cyclic_Iterative_Adversarial_Robustness_Distillation_ICCV_2025_paper.pdf), Tables5/6 CIFAR-100 **Robust** columns for black-box results. Values are percentages; parentheses are result minus the same-model baseline in pp. Means and differences are calculated before rounding. The supplementary AA table is CIFAR-10 and supplies no CIFAR-100 baseline.

### ResNet-18 / CIFAR-100

| Metric | Paper CIARD baseline | 0914v1 C100-R1 (Δ vs baseline) | 0914v1 C100-R2 (Δ vs baseline) | 0918v1 C100-E2 (Δ vs baseline) | 1004 H2 (Δ vs baseline) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 65.73 | 63.53 (-2.20 pp) | 63.78 (-1.95 pp) | 65.83 (+0.10 pp) | 65.96 (+0.23 pp) |
| White-box FGSM | 34.47 | 35.37 (+0.90 pp) | 35.70 (+1.23 pp) | 33.78 (-0.69 pp) | 34.52 (+0.05 pp) |
| White-box PGDsat | 28.05 | 30.91 (+2.86 pp) | 31.20 (+3.15 pp) | 28.22 (+0.17 pp) | 29.76 (+1.71 pp) |
| White-box PGDtrades | 29.45 | 32.02 (+2.57 pp) | 32.28 (+2.83 pp) | 29.64 (+0.19 pp) | 30.78 (+1.33 pp) |
| White-box CW | 24.43 | 27.72 (+3.29 pp) | 27.82 (+3.39 pp) | 26.46 (+2.03 pp) | 26.66 (+2.23 pp) |
| Black-box PGDtrades | 42.29 | 43.35 (+1.06 pp) | 43.29 (+1.00 pp) | 43.29 (+1.00 pp) | 43.52 (+1.23 pp) |
| Square (query-based) | 49.76 | 51.65 (+1.89 pp) | 51.69 (+1.93 pp) | 52.50 (+2.74 pp) | 51.90 (+2.14 pp) |
| Black-box CW | 41.44 | 42.27 (+0.83 pp) | 42.49 (+1.05 pp) | 42.23 (+0.79 pp) | 42.29 (+0.85 pp) |
| Seven-attack mean | 35.70 | 37.61 (+1.91 pp) | 37.78 (+2.08 pp) | 36.59 (+0.89 pp) | 37.06 (+1.36 pp) |
| Eight-metric mean | 39.45 | 40.85 (+1.40 pp) | 41.03 (+1.58 pp) | 40.24 (+0.79 pp) | 40.67 (+1.22 pp) |

**H2 is now the active CIFAR-100 ResNet source under `CIARD_Expansion_resnet18_cifar100`, with a matching `best_backup/resnet18_cifar100_1004_h2` snapshot.** Its epoch190 EMA is bound to training137534/evaluation137729 and checkpoint SHA256 `e32461425fc112b763c6650c65756979e8ca75c3342119acbe4363075be0d13b`. It exceeds all eight paper values under the retained protocol; the smallest margin is FGSM +0.05pp, or five correct predictions above the paper count. **H2 AutoAttack is24.22%**, reported separately without a CIFAR-100 paper threshold. This single-seed result does not establish stable superiority. Full local evidence: `结果分析/1004-cifar100-binary-v1_结果分析.md` and `run/1004-cifar100-binary-v1/resnet18_cifar100_nb_l2_w40_s0/logs/eval_stdout_137729.log`; these original experiment outputs are not bundled in this source package.

**Historical R1/R2 results and all their table cells are retained unchanged.** R1 source remains in `origin_code/0914v1` and its original run. The previous package R2 source remains frozen at `origin_code/0917v1-cifar10/IJCV_KD/CIARD_Expansion_resnet18_cifar100` and in its original run, with a recorded mapping for this local replacement. R2−R1 in table order is +0.25 / +0.33 / +0.29 / +0.26 / +0.10 / −0.06 / +0.04 / +0.22 pp: seven primary metrics rise and black-box PGDtrades falls. AutoAttack is **25.70% for R1 and 25.59% for R2**. The higher R2 eight-metric mean does not imply improvement on every metric; neither run exceeds the paper's Clean (R1/R2 gaps: 2.20/1.95 pp). These are single-seed observations, not statistically established superiority; their historical configurations and evidence remain recorded in the manifest.

**0918v1 C100-E2 remains a historical result-only reference alongside R2; the current local active entry is H2.** E2 uses the epoch-190 EMA `student_best.pth`, evaluated on all 10,000 test images; training/evaluation jobs **134960 / 135316** both completed successfully. Its original local experiment directory is `/home/lixidong25/mycode/CIARD_Expansion/run/0918v1/resnet18_cifar100_ce0p05_s120_w80_tau2p0_tm0p011316/` (a local filesystem path, not a directory included in this GitHub repository). The evaluation log is `logs/eval_best_stdout_135316.log` within that directory, and the complete local report is `结果分析/0918v1_cifar100_结果分析.md`. E2 source code, checkpoints and logs are not synchronized here.

E2 improves Clean by **2.05 pp** relative to R2, while FGSM falls **1.92 pp**; its AutoAttack is **24.06%**, versus R2's 25.59%. Relative to the paper baseline used in the new column, E2 exceeds seven of eight primary metrics: Clean is +0.10 pp, but FGSM remains −0.69 pp. R2 and E2 therefore remain complementary references, not an all-metric replacement of one by the other. E2 trained and was evaluated on one A800; historical R2 trained on two 4090s, so their comparison is not a strict single-factor experiment.

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

The historical complete report is `结果分析/0914v1_cifar100_结果分析.md`; all four0914 configurations and eight successful training/evaluation jobs remain recorded. Source/result/log/checkpoint hashes, 100-class CPU strict loading and full10,000-image log/JSON/count agreement were verified. Each result uses one EMA checkpoint, without combining best individual metrics across checkpoints. **The current CIFAR-100 source entries are H2 and the unchanged0914 M1.** R1/R2/M2 retain their original independent runs and historical manifest records. Prior R1/R2/M1 publication belongs to `origin_code/0914v1`, which is unchanged. Historical filenames retain `eval_best_0909v1_<job>.json`; H2 instead uses `eval_137729.json` and a fixed final190EMA. H2 source/script/manifest bindings, actual EMA state fingerprints, complete nine-metric counts and successful jobs137534/137729 passed the frozen verifier; its report is `结果分析/1004-cifar100-binary-v1_结果分析.md`. Previous publication/verification records remain available; the2026-10-05 H2 source update is authorized for GitHub publication. The retained synchronization verification describes the earlier local-only checkpoint; remote publication evidence is recorded separately.

The historical0914 CIFAR-100 runs retain test-loader `(Clean+PGD proxy)/2` selection of EMA best, introducing selection bias, and their stochastic attacks were not all explicitly seeded. H2 uses fixed final190EMA without test-loader checkpoint selection and fixes each metric's attack seed to0; historical test results still informed its design. The retained evaluation uses L-infinity8/255, PGDsat20 steps/2/255, PGDtrades20 steps/.003, CW30 steps/2/255 and Square100queries. Black-box PGDtrades/CW transfer from WRN-70-16; Square queries the student. The official/historical PGDtrades evaluator's .003 differs from the paper text's2/255, while Square100queries matches the stated paper budget; training remains PGD-10 at2/255. These are single-seed numerical comparisons to published values under the project's frozen protocol, not proof of exactly matched reproduction, multi-seed stability or dominance over all historical models. Mean accuracy is not joint worst-case accuracy. Only the user submits GPU jobs.

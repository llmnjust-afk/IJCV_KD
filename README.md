# IJCV_KD candidate source

The active ResNet-18 source is the **0906v2 split-mixing candidate**
`resnet18_split_t025_n020_s120_w40_p081740`, developed from the completed
0906v1 G3 experiment. It uses target-class mixing `0.25`, non-target mixing
`0.20`, and the G3 reference mixing/mass weight `0.20`; start `120`, warmup
`40`, push `0.081740`. It was selected before evaluation. The completed 0906v2 G3 evaluation now
reports Clean 88.70, FGSM 61.18 and an eight-metric mean of 64.87%; it did
not establish this active configuration as the batch winner. The active
source remains in place; the separately archived G7 below is the user's
selected seven-metric candidate.

See [the ResNet README](CIARD_Expansion_resnet18_cifar10/README.md) for the method,
fixed configuration, source provenance, and manual preparation.

## Completed 0906v2 G7 backup

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
are uploaded. The existing active ResNet source and earlier backups are
preserved. G7 uses the same historical protocol limitations described below.

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
not to the new 0906v2 candidate.

Both versions use the historical 50,000-image training protocol and test-loader
checkpoint selection, which introduces selection bias. Stochastic attack seeds
are not all explicitly fixed. PGDtrades uses step size 0.003 in the frozen
evaluator; the paper describes 2/255, so comparison to its published numbers
is not proof of an exactly matched reproduction.

## Other source and execution notes

`CIARD_Expansion_mobilenetv2_cifar10` retains the 0903 push=0.075 candidate.
Its completed evaluation missed the CIARD baseline on Clean (89.07 versus
89.51) and black-box CW (65.29 versus 66.12); the verified MobileNet reference
remains the 0624 source. MobileNet code and the earlier `best_backup/` entries are unchanged; G7 is an additional backup.

Both models use raw-input WRN-34-10 robust and ResNet-56 natural teachers.
ResNet has 4090 Slurm wrappers and completion/hash checks; MobileNet retains
its historical 3090 wrappers. All training and evaluation jobs must be
submitted manually by the user. Data, weights, outputs, logs and local resource
links are excluded. Historical 0830 SARD material is not the current execution
interface.

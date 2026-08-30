# SARD: Strength-Adaptive Reliability-Calibrated Distillation

IJCV journal extension of the ICCV 2025 paper **CIARD** (Cyclic Iterative Adversarial Robustness Distillation).

SARD introduces two modules on top of CIARD:
1. **SAA** (Strength-Adaptive Attack) — Beta-distribution epsilon sampling with curriculum, replacing fixed-epsilon adversarial example generation
2. **RCD** (Reliability-Calibrated Distillation) — Per-sample Teacher Reliability Score (TRS) weighting that down-weights distillation from unreliable teacher predictions

The MobileNet-V2 and ResNet-18 directories now expose the same SARD switches
and core fixes. Their student architectures and architecture-specific baseline
parameters remain independent.

## Source status

This directory is a repaired source snapshot, not an independent experiment
directory and not a submission target. Its historical Slurm files still refer
to old locations. Before a formal run, copy the selected student source into a
new, empty run directory, link that directory's `data` and `models` to the
project-wide read-only resources, fix the configuration and unique prefix in
that copy, and prepare matching training/evaluation scripts.

The source deliberately does not select a physical GPU. It respects the
`CUDA_VISIBLE_DEVICES` allocation supplied by Slurm and discards the legacy
`CIARD_GPU`, `CIARD_STUDENT`, and `CIARD_PREFIX` variables.

Both training files expose `--sard_saa`, `--sard_rcd`, `--epochs`, `--prefix`,
and `--original_ciard` for source-level diagnostics. Formal independent
variants must still write the selected settings directly into their copied
`CIARD.py`, use a unique initially empty prefix, and print the resolved
configuration from Python. The four conceptual SARD controls are baseline,
SAA-only, RCD-only, and SAA+RCD; the actual 0830 experiment matrix has not yet
been selected or prepared.

## Evaluation

### Fast Evaluation (white-box + black-box, ~2 min)

```bash
python fast_eval.py --checkpoint model/sard_200ep/student_best.pth --prefix sard_200ep
```

Runs: Clean Accuracy, WB PGD-20 (two step sizes), WB FGSM, WB CW L-inf, BB PGD-20, BB CW L-inf.
Outputs structured JSON to `fast_eval_<prefix>_<job-or-time>.json` and prints
`EVAL_COMPLETE` only after all required metrics are present.

### Full Evaluation (includes AutoAttack, ~30 min)

Edit `attack_eval.py` to set the checkpoint path, then:

```bash
python attack_eval.py
```

## Teacher Models

| Teacher | Architecture | Source | Clean Acc | Role |
|---------|-------------|--------|-----------|------|
| Robust teacher | WRN-34-10 (raw) | Project shared TRADES checkpoint | ~85% | Adversarial distillation |
| Natural teacher | ResNet-56 (raw) | Project shared checkpoint | ~93% | Clean distillation |

The paper uses this same teacher pair for both MobileNet-V2 and ResNet-18
students on CIFAR-10. `setup_models.sh` now performs strict CPU verification
only; it never downloads or overwrites the project-wide checkpoints.

## CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--sard_saa` | int | 1 | Enable SAA module (0=off, 1=on) |
| `--sard_rcd` | int | 1 | Enable RCD module (0=off, 1=on) |
| `--epochs` | int | 300 | Total training epochs |
| `--prefix` | str | `Cifar10_MobileNetV2_tm010_repeat0620` | Model save directory name |
| `--original_ciard` | flag | off | Disable ALL non-original-CIARD modifications for true single-variable ablation |

## Checkpoint Output

Training saves checkpoints to `model/<prefix>/`:

| File | Save Condition | Content |
|------|----------------|---------|
| `student_<epoch>.pth` | Every `epochs//6` epochs | Student model + optimizer + epoch |
| `student_latest.pth` | Last 17% of epochs (every epoch) | Student model + optimizer + epoch |
| `student_best.pth` | When (clean+robust)/2 improves on **validation set** | Best student model |

> Checkpoint selection uses a 5k validation split from the training set, NOT the test set, to avoid test set leakage.

## Key Bug Fixes (vs previous version)

### Critical Fixes

1. **Adaptive Temperature per-batch multiply (FIXED)**
   - `temp_scale` was computed inside the batch loop, multiplying `temp_adv` ~391 times per epoch
   - Temperature exploded to `temp_max=10` within 5 batches, destroying KD signal
   - Fix: compute one epoch scale and apply it to non-persistent effective temperatures; the base temperatures are never multiplied per batch

2. **KL temperature asymmetry (FIXED)**
   - Only teacher logits were divided by temperature; student logits were not
   - Fix: both teacher and student logits now divided by the same temperature T

3. **RCD floor ineffective on wrong samples (FIXED)**
   - Formula was `correct * margin_gate * (floor + ...)` — when `correct=0`, TRS=0 regardless of floor
   - Fix: restructured to `floor + correct * (full_weight - floor)`, so wrong predictions get `floor` minimum

4. **Baseline not single-variable (FIXED)**
   - `--sard_saa 0 --sard_rcd 0` still had label smoothing, adaptive temp, FGSM anchor, etc.
   - Fix: `--original_ciard` flag disables all non-original modifications

5. **Teacher epoch>50 hardcoded (FIXED)**
   - Hardcoded `if epoch > 50` meant 60-epoch experiments only updated teacher in last 10 epochs
   - Fix: proportional scaling `teacher_update_epoch = int(epochs * 50/300)`

### Moderate Fixes

6. **Teacher BN not frozen (FIXED)** — teacher stayed in `train()` mode even when `teacher_lr=0`, corrupting BatchNorm statistics. Now switched to `eval()` when not updating.

7. **Extra forward pass removed (FIXED)** — `adv_teacher_nat = teacher(train_batch_data)` was computed but never used in any loss.

8. **SAA step_size confound (FIXED)** — SAA changed both epsilon distribution AND step size (`2*eps/10`). Now uses fixed `2/255` step size, isolating the epsilon curriculum effect.

9. **teacher-margin conflict dead code (FIXED)** — `teacher_margin_conflict_scale` was computed but never applied to loss. Now applied when batch-level gate is active.

10. **Gradient sign error (FIXED)** — Per-sample conflict check used `kd_grad_true > 0 AND tm_grad_true < 0` as "agree", but in gradient descent these are opposite directions. Fixed to use `(kd_grad_true * tm_grad_true) > 0`.

11. **Test set leakage (FIXED)** — Checkpoint selection now uses 5k validation split from training set instead of test set.

12. **PyTorch version compatibility (FIXED)** — Added `safe_torch_load()` around all training, evaluation, and weight-averaging loads, including support for the verified PyTorch 1.10 environment.

13. **Teacher setup safety (FIXED)**:
    - Default training strict-loads the existing raw WRN-34-10 and ResNet-56 checkpoints.
    - The incompatible WRN-34-20 converter and generic teacher-training entrypoint are disabled.
    - `setup_models.sh` is verification-only and cannot overwrite shared weights.

14. **Eval bugs (FIXED)**:
    - CW attack: clamp `input + perturbation` to [0,1] inside loop, not just at end
    - PGD/FGSM: call `model.zero_grad()` to prevent parameter gradient accumulation
    - `parse_known_args` → `parse_args` to catch typos
    - Structured JSON output with `EVAL_COMPLETE` marker

## SARD Method Details

### SAA: Strength-Adaptive Attack

Instead of a fixed epsilon=8/255, SAA samples epsilon from a Beta distribution with a curriculum that shifts the distribution mean from ~0.29*eps_max (early) to ~0.71*eps_max (late). The step size is kept fixed at 2/255 to isolate the epsilon curriculum effect.

### RCD: Reliability-Calibrated Distillation

The robust teacher (~58% robust accuracy) produces wrong predictions on ~42% of adversarial examples. RCD computes a per-sample TRS:

- When teacher is **correct**: `TRS = floor + (1-floor) * confidence * margin_gate` (full weight)
- When teacher is **wrong**: `TRS = floor` (minimum signal preserved, floor=0.1)

This ensures the floor always protects against complete signal loss, even on misclassified samples.

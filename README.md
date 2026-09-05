# IJCV_KD candidate source

The active CIFAR-10 directories contain:

- `CIARD_Expansion_resnet18_cifar10`: the selected 0906 target-mixing candidate
  `resnet18_tmix_a020_s120_w40_p081740`, with alpha `0.20`, start `120`, warmup
  `40`, and push `0.081740`. No completed 0906 evaluation was available at
  selection: this is an **unvalidated candidate**, with no new-best or
  all-metrics improvement claim.
- `CIARD_Expansion_mobilenetv2_cifar10`: the retained 0903 candidate, based on
  the 0624 mainline with push changed from `0.05` to `0.075`. Its completed 0903
  evaluation improved white-box metrics but missed the CIARD baseline on
  Clean (`89.07` versus `89.51`) and black-box CW (`65.29` versus `66.12`).
  It does not replace the verified MobileNet-V2 best source.

The 0903 batch completed training and evaluation. The former ResNet projected
clean/FGSM auxiliary candidate did not achieve all-metrics improvement and is
replaced here by the 0906 candidate. Alpha `0.20` is the middle mixing strength
of the prepared grid and preserves the original push strength. This selection
is an engineering choice, not a ranking inferred from incomplete training logs.

Both models use raw-input WRN-34-10 robust and ResNet-56 natural teachers.
Training retains 50,000 training examples and test-loader checkpoint selection,
which introduces test-selection bias. Evaluation retains the historical attack
protocol, whose stochastic attack seeds are not all explicitly fixed. The new
ResNet log checker verifies nine metrics and writes historical-protocol JSON;
it does not make evaluation deterministic or change attack budgets.

See the model READMEs for preparation. ResNet has matching 4090 Slurm wrappers;
MobileNet retains its historical 3090 wrappers. All Slurm training and evaluation
jobs must be submitted manually by the user. Datasets, teacher weights,
checkpoints, logs, and local resource links are excluded from this publication.

`best_backup/` remains the frozen record of the previously verified best source.
The 0830 SARD guides and supporting material are historical, not the current
execution interface. This update changes the ResNet candidate and documentation;
it does not promote a new verified best model.

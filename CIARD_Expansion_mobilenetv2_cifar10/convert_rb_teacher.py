"""Deprecated WRN-34-20 conversion entrypoint.

The CIFAR-10 CIARD protocol in this project uses the shared raw WRN-34-10
checkpoint. Converting Rice2020 WRN-34-20 into the generic public checkpoint
name would silently change the teacher architecture for every experiment.
"""

raise SystemExit(
    "Disabled: this project uses models/model_cifar_wrn.pt as raw WRN-34-10. "
    "Do not overwrite the shared checkpoint with a WRN-34-20 conversion."
)

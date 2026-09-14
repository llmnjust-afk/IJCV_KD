"""Disabled teacher-training entrypoint for the shared-weight project tree."""

raise SystemExit(
    "Disabled: teacher generation must write an architecture-specific new file "
    "and manifest outside the shared models symlink. Existing CIARD runs use "
    "the verified raw WRN-34-10 checkpoint."
)

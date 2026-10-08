#!/bin/bash
# Verification-only helper for the shared CIFAR-10 teacher checkpoints.
# It intentionally never downloads, converts, trains, or overwrites weights.

set -euo pipefail

python3 -c '
import hashlib
import os
import torch
from cifar10_models import wideresnet
from cifar10_nat_teacher_models import cifar10_resnet56

def safe_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

checks = (
    ("models/model_cifar_wrn.pt", wideresnet(), "WRN-34-10 raw"),
    ("models/nat_teacher_checkpoint/cifar10_resnnet56.pth", cifar10_resnet56(), "ResNet-56 raw"),
)
for path, model, architecture in checks:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    state_dict = {key.replace("module.", ""): value for key, value in safe_load(path).items()}
    model.load_state_dict(state_dict, strict=True)
    print("{} strict-load OK".format(architecture))
    print("  path={}".format(os.path.realpath(path)))
    print("  size_bytes={}".format(os.path.getsize(path)))
    print("  sha256={}".format(sha256(path)))
'

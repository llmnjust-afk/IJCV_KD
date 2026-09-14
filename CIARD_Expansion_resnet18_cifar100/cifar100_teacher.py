"""CIFAR-100 teachers; raw [0, 1] input is normalized inside each model."""
import torch.nn.functional as F
from robustbench.model_zoo.architectures.dm_wide_resnet import (
    CIFAR100_MEAN, CIFAR100_STD, DMWideResNet, Swish)


class FeatureDMWideResNet(DMWideResNet):
    """Historical CIFAR-100 natural-teacher feature interface."""

    @property
    def feature_dim(self):
        return self.num_channels

    def forward(self, x, return_feature=False):
        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4)
        out = (x - self.mean) / self.std
        out = self.init_conv(out)
        out = self.layer(out)
        out = self.relu(self.batchnorm(out))
        out = F.avg_pool2d(out, 8)
        feat = out.view(-1, self.num_channels)
        logits = self.logits(feat)
        if return_feature:
            return logits, feat
        return logits


def robust_teacher():
    return DMWideResNet(num_classes=100, depth=70, width=16,
                       activation_fn=Swish, mean=CIFAR100_MEAN, std=CIFAR100_STD)


def natural_teacher():
    return FeatureDMWideResNet(num_classes=100, depth=22, width=6,
                              activation_fn=Swish, mean=CIFAR100_MEAN, std=CIFAR100_STD)

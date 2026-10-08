"""Keep torchvision's training data path without checking the test file."""
from torchvision.datasets import CIFAR100


class CIFAR100TrainOnly(CIFAR100):
    # torchvision 0.11 otherwise hashes both splits even with train=True.
    test_list = []

    def __init__(self, root, transform=None):
        super().__init__(root=root, train=True, transform=transform, download=False)

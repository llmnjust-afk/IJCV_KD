"""Explicit paired seed for every full-test metric, without changing attacks."""
import random
import numpy as np
import torch


def reset_eval_seed(metric):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print('EVAL_SEED metric={} seed=0 cudnn_deterministic=True benchmark=False'.format(metric), flush=True)

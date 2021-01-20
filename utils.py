import torch
import numpy as np


def setSeed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
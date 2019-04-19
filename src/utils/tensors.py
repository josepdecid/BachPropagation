from typing import Tuple

import torch
from torch import cuda, device as dev

use_cuda = cuda.is_available()
device = dev('cuda' if use_cuda else 'cpu')


def zeros_target(dims: Tuple):
    return torch.zeros(size=dims, device=device)


def ones_target(dims: Tuple):
    return torch.ones(size=dims, device=device)

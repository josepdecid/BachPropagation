import torch
import numpy as np
from typing import IO

File = IO[str]
IntTensor = torch.IntTensor
FloatTensor = torch.FloatTensor
NDArray = np.ndarray

Optimizer = torch.optim.Optimizer
Scheduler = torch.optim.lr_scheduler
Criterion = torch.nn.modules.loss.CrossEntropyLoss
NNet = torch.nn.Module

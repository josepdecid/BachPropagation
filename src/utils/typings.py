import torch
from typing import IO

File = IO[str]
IntTensor = torch.IntTensor
FloatTensor = torch.FloatTensor

Optimizer = torch.optim.Optimizer
Scheduler = torch.optim.lr_scheduler
Criterion = torch.nn.modules.loss.CrossEntropyLoss
NNet = torch.nn.Module

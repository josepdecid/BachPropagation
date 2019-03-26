import torch
from typing import TextIO

File = TextIO
IntTensor = torch.IntTensor
FloatTensor = torch.FloatTensor

Optimizer = torch.optim.Optimizer
Criterion = torch.nn.modules.loss.CrossEntropyLoss
NNet = torch.nn.Module
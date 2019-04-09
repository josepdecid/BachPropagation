import logging
import sys

import torch

from dataset.MusicDataset import MusicDataset
from model.GANModel import GANModel
from model.train import Trainer
from utils.tensors import device

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    logging.info(f'Running Python {sys.version.split()[0]} with PyTorch {torch.__version__} in {device}')

    model = GANModel()
    dataset = MusicDataset()

    trainer = Trainer(model=model, dataset=dataset)
    trainer.train()

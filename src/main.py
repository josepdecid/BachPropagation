import argparse
import logging
import sys

import torch

from dataset.MusicDataset import MusicDataset
from model.GANModel import GANModel
from model.train import Trainer
from constants import FLAGS
from utils.tensors import device


def set_flags(flags):
    FLAGS['viz'] = flags.viz


def run_model():
    model = GANModel()
    dataset = MusicDataset()

    trainer = Trainer(model=model, dataset=dataset)
    trainer.train()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    logging.info(f'Running Python {sys.version.split()[0]} with PyTorch {torch.__version__} in {device}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true')
    args = parser.parse_args()

    set_flags(args)
    run_model()

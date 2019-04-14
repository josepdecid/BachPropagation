import logging

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from constants import LR_G, L2_G, L2_D, LR_D, LR_PAT_D, LR_PAT_G
from model.gan.GANDiscriminator import GANDiscriminator
from model.gan.GANGenerator import GANGenerator
from utils.tensors import device
from utils.typings import NNet, Optimizer, Criterion, Scheduler


class GANModel:
    def __init__(self):
        logging.info('Creating GAN model...')

        self.generator: NNet = GANGenerator().to(device)
        self.g_optimizer: Optimizer = torch.optim.Adam(self.generator.parameters(), lr=LR_G, weight_decay=L2_G)
        self.g_scheduler: Scheduler = ReduceLROnPlateau(self.g_optimizer, mode='min', patience=LR_PAT_G)

        self.discriminator: NNet = GANDiscriminator().to(device)
        self.d_optimizer: Optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=LR_D, weight_decay=L2_D)
        self.d_scheduler: Scheduler = ReduceLROnPlateau(self.d_optimizer, mode='min', patience=LR_PAT_D)

        self.training_criterion: Criterion = nn.BCELoss()
        self.pretraining_criterion: Criterion = nn.CrossEntropyLoss()
        self.pretraining_discriminator_criterion: Criterion = nn.MSELoss()

    def train_mode(self):
        self.generator.train()
        self.discriminator.train()

    def eval_mode(self):
        self.generator.eval()
        self.discriminator.eval()

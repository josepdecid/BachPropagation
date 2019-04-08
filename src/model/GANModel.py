import logging

import torch

from model.GANDiscriminator import GANDiscriminator
from model.GANGenerator import GANGenerator
from utils.constants import LR_G, L2_G, L2_D, LR_D
from utils.tensors import device
from utils.typings import NNet, Optimizer, Criterion, FloatTensor


class GANModel:
    def __init__(self):
        logging.info('Creating GAN model...')

        self.generator: NNet = GANGenerator().to(device)
        self.g_optimizer: Optimizer = torch.optim.Adam(self.generator.parameters(), lr=LR_G, weight_decay=L2_G)
        self.g_criterion: Criterion = GANModel._generator_criterion

        self.discriminator: NNet = GANDiscriminator().to(device)
        self.d_optimizer: Optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=LR_D, weight_decay=L2_D)
        self.d_criterion: Criterion = GANModel._discriminator_criterion

        # with SummaryWriter(log_dir=f'{PROJECT_PATH}/res/log', comment='Generator') as w:
        #     h = self.generator(Variable(torch.zeros(BATCH_SIZE, 42, MAX_POLYPHONY), requires_grad=True))
        #     w.add_graph(self.generator, h)

        # with SummaryWriter(log_dir=f'{PROJECT_PATH}/res/log', comment='Discriminator') as w:
        #     h = self.discriminator(Variable(torch.zeros(BATCH_SIZE, 42, MAX_POLYPHONY), requires_grad=True))
        #     w.add_graph(self.discriminator, h)

    # PyTorch working modes

    def train_mode(self):
        self.generator.train()
        self.discriminator.train()

    def eval_mode(self):
        self.generator.eval()
        self.discriminator.eval()

    # Loss functions

    @staticmethod
    def _generator_criterion(d_g_z: FloatTensor) -> FloatTensor:
        """
        Loss function for Generator G.
        Calculates 1/m · ∑ log(1 - D(G(z))
        where *z* is the uniform random vector (noise) ∈ [0, 1]^T
        :param d_g_z: Tensor corresponding to the discriminator prediction D(G(z))
        :return: Loss of G
        """
        return torch.mean(torch.log(1 - d_g_z))

    @staticmethod
    def _discriminator_criterion(d_x: FloatTensor, d_g_z: FloatTensor) -> FloatTensor:
        """
        Loss function for Discriminator D.
        Calculates 1/m · ∑ -log(D(x)) - log(1 - D(G(z))
        where *z* is the uniform random vector (noise) ∈ [0, 1]^T
        and *x* is the sequence of real training data
        :param d_x: Tensor corresponding to the discriminator real prediction D(x)
        :param d_g_z: Tensor corresponding to the discriminator fake prediction D(G(z))
        :return: Loss of D
        """
        return torch.mean(torch.log(d_x) + torch.log(1 - d_g_z))

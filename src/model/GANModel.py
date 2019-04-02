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
    def _generator_criterion(predictions: FloatTensor) -> FloatTensor:
        """
        Loss function for Generator G.
        Calculates 1/m · ∑ log(1 - D(G(z))
        where *z* is the uniform random vector (noise) ∈ [0, 1]^T
        :param predictions: Tensor corresponding to the discriminator prediction D(G(z))
        :return: Loss of G
        """
        return torch.mean(torch.log(1 - predictions))

    @staticmethod
    def _discriminator_criterion(real_predictions: FloatTensor, fake_predictions: FloatTensor) -> FloatTensor:
        """
        Loss function for Discriminator D.
        Calculates 1/m · ∑ -log(D(x)) - log(1 - D(G(z))
        where *z* is the uniform random vector (noise) ∈ [0, 1]^T
        and *x* is the sequence of real training data
        :param real_predictions: Tensor corresponding to the discriminator real prediction D(G(z))
        :param fake_predictions: Tensor corresponding to the discriminator fake prediction D(x)
        :return: Loss of D
        """
        real_data_loss = -torch.log(real_predictions)
        fake_data_loss = torch.log(1 - fake_predictions)
        return torch.mean(real_data_loss - fake_data_loss)

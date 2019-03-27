import torch
from torch.nn import functional as F

from model.GANDiscriminator import GANDiscriminator
from model.GANGenerator import GANGenerator
from utils.typings import NNet, Optimizer, Criterion


class GANModel:
    def __init__(self):
        self.generator: NNet = None
        self.g_optimizer: Optimizer = None
        self.g_criterion: Criterion = None

        self.discriminator: NNet = None
        self.d_optimizer: Optimizer = None
        self.d_criterion: Criterion = None

    def initialize_generator(self, hid_dim: int, optimizer: Optimizer):
        self.generator = GANGenerator(layers=2, hid_dim=hid_dim)
        self.g_optimizer = optimizer(self.generator.parameters())
        self.g_criterion = GANModel._generator_criterion

    def initialize_discriminator(self, hid_dim: int, optimizer: Optimizer):
        self.discriminator = GANDiscriminator(layers=2, hid_dim=hid_dim)
        self.d_optimizer = optimizer(self.discriminator.parameters())
        self.d_criterion = GANModel._discriminator_criterion

    # PyTorch working modes

    def train_mode(self):
        self.generator.train()
        self.discriminator.train()

    def test_mode(self):
        self.generator.eval()
        self.discriminator.eval()

    # Loss functions

    @staticmethod
    def _generator_criterion(prediction, ones):
        return torch.mean(torch.log(ones - prediction))

    @staticmethod
    def _discriminator_criterion(y, y_hat):
        return F.binary_cross_entropy(y, y_hat)

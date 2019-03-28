import torch
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn import functional as F

from model.GANDiscriminator import GANDiscriminator
from model.GANGenerator import GANGenerator
from utils.constants import PROJECT_PATH
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

        with SummaryWriter(log_dir=f'{PROJECT_PATH}/res/log', comment='Generator') as w:
            h = self.generator(Variable(torch.zeros(1, 1, 1), requires_grad=True))
            w.add_graph(self.generator, h, verbose=True)

    def initialize_discriminator(self, hid_dim: int, optimizer: Optimizer):
        self.discriminator = GANDiscriminator(layers=2, hid_dim=hid_dim)
        self.d_optimizer = optimizer(self.discriminator.parameters())
        self.d_criterion = GANModel._discriminator_criterion

        with SummaryWriter(log_dir=f'{PROJECT_PATH}/res/log', comment='Discriminator') as w:
            h = self.discriminator(Variable(torch.zeros(1, 1, 1), requires_grad=True))
            w.add_graph(self.discriminator, h, verbose=True)

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

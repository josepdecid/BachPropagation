import torch
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn import functional as F

from model.GANDiscriminator import GANDiscriminator
from model.GANGenerator import GANGenerator
from utils.constants import PROJECT_PATH, LAYERS_G, HIDDEN_DIM_G, LAYERS_D, HIDDEN_DIM_D
from utils.typings import NNet, Optimizer, Criterion


class GANModel:
    def __init__(self):
        self.generator: NNet = None
        self.g_optimizer: Optimizer = None
        self.g_criterion: Criterion = None
        self._initialize_generator()

        self.discriminator: NNet = None
        self.d_optimizer: Optimizer = None
        self.d_criterion: Criterion = None
        self._initialize_discriminator()

    def _initialize_generator(self):
        self.generator = GANGenerator(layers=LAYERS_G, hid_dim=HIDDEN_DIM_G)
        self.g_optimizer = torch.optim.Adam(self.generator.parameters())
        self.g_criterion = GANModel._generator_criterion

        with SummaryWriter(log_dir=f'{PROJECT_PATH}/res/log', comment='Generator') as w:
            h = self.generator(Variable(torch.zeros(1, 1, 1), requires_grad=True))
            w.add_graph(self.generator, h)

    def _initialize_discriminator(self):
        self.discriminator = GANDiscriminator(layers=LAYERS_D, hid_dim=HIDDEN_DIM_D)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters())
        self.d_criterion = GANModel._discriminator_criterion

        with SummaryWriter(log_dir=f'{PROJECT_PATH}/res/log', comment='Discriminator') as w:
            h = self.discriminator(Variable(torch.zeros(1, 1, 1), requires_grad=True))
            w.add_graph(self.discriminator, h)

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

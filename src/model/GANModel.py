from torch import nn

from model.utils.GANDiscriminator import GANDiscriminator
from model.utils.GANGenerator import GANGenerator


class GANModel(nn.Module):
    def __init__(self):
        super(GANModel, self).__init__()

        self.generator = GANGenerator()
        self.discriminator = GANDiscriminator()

    def forward(self, x):
        x = self.generator(x)
        x = self.discriminator(x)
        return x

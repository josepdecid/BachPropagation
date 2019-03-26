import torch
from torch import nn
from torch.autograd import Variable
from utils.constants import MAX_POLYPHONY

from model.RNN import RNN


class GANGenerator(nn.Module):
    def __init__(self, hid_dim: int):
        """
        **Generator** network of a GAN model.
        Applies a RNN to a random noise generator, and passes each hidden state through a Dense Layer.
        Tries to generate realistic synthetic data to trick the **Discriminator** to make it say that it's real.
        :param hid_dim: The number of features in the hidden state *h* of the RNN.
        """
        super(GANGenerator, self).__init__()

        self.rnn = RNN(architecture='GRU', inp_dim=MAX_POLYPHONY, hid_dim=hid_dim, layers=2, bidirectional=False)
        self.dense = nn.Linear(in_features=hid_dim, out_features=hid_dim)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.dense(x)
        return x

    @staticmethod
    def noise(size):
        """
        Generates a 1-d vector of gaussian sampled random values.
        :param size: Number of features (dimension) of the data.
        """
        return Variable(torch.randn(size, 100))

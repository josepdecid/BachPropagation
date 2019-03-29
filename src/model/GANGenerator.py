from typing import Tuple

import torch
from torch import nn
from torch.autograd import Variable

from model.RNN import RNN
from utils.constants import MAX_POLYPHONY, HIDDEN_DIM_G, LAYERS_G, BIDIRECTIONAL_G


class GANGenerator(nn.Module):
    def __init__(self):
        """
        **Generator** network of a GAN model.
        Applies a RNN to a random noise generator, and passes each hidden state through a Dense Layer.
        Tries to generate realistic synthetic data to trick the **Discriminator** to make it say that it's real.
        """
        super(GANGenerator, self).__init__()

        self.rnn = RNN(architecture='GRU',
                       inp_dim=MAX_POLYPHONY,
                       hid_dim=HIDDEN_DIM_G,
                       layers=LAYERS_G,
                       bidirectional=BIDIRECTIONAL_G)

        dense_input_features = (2 if BIDIRECTIONAL_G else 1) * HIDDEN_DIM_G
        self.dense = nn.Linear(in_features=dense_input_features, out_features=MAX_POLYPHONY)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.dense(x)
        return x

    @staticmethod
    def noise(dims: Tuple):
        """
        Generates a 2-d vector of gaussian sampled random values.
        :param dims: Tuple with the dimensions of the data.
        """
        return Variable(torch.randn(dims))

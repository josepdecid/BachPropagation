from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from constants import HIDDEN_DIM_G, LAYERS_G, BIDIRECTIONAL_G, TYPE_G, INPUT_FEATURES, MIN_FREQ_NOTE, MAX_FREQ_NOTE, \
    MAX_VELOCITY
from model.gan.RNN import RNN
from utils.tensors import device


class GANGenerator(nn.Module):
    def __init__(self):
        """
        **Generator** network of a GAN model.
        Applies a RNN to a random noise generator, and passes each hidden state through a Dense Layer.
        Tries to generate realistic synthetic data to trick the **Discriminator** to make it say that it's real.
        """
        super(GANGenerator, self).__init__()

        self.rnn = RNN(architecture=TYPE_G,
                       inp_dim=INPUT_FEATURES,
                       hid_dim=HIDDEN_DIM_G,
                       layers=LAYERS_G,
                       bidirectional=BIDIRECTIONAL_G).to(device)

        self.dense_input_features = (2 if BIDIRECTIONAL_G else 1) * HIDDEN_DIM_G
        self.dense_1 = nn.Linear(in_features=self.dense_input_features, out_features=INPUT_FEATURES)

    def forward(self, x, pretraining=False):
        out, (h_n, c_n) = self.rnn(x)
        if pretraining:
            y = h_n[(-2 if BIDIRECTIONAL_G else -1):].view(-1, self.dense_input_features)
            y = self.dense_1(y)
        else:
            y = self.dense_1(out)
        y = F.relu(y)
        return y

    @staticmethod
    def noise(dims: Tuple):
        """
        Generates a 2-d vector of uniform sampled random values.
        :param dims: Tuple with the dimensions of the data.
        """
        notes = (MIN_FREQ_NOTE - MAX_FREQ_NOTE) * torch.rand(dims) + MAX_FREQ_NOTE
        velocity = MAX_VELOCITY * torch.rand(dims)
        duration = torch.tensor(100) * torch.rand(dims)
        since_previous = torch.tensor(100) * torch.rand(dims)
        return torch.stack((notes, velocity, duration, since_previous), dim=2).to(device)

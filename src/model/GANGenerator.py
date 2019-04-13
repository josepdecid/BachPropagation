from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from model.RNN import RNN
from constants import HIDDEN_DIM_G, LAYERS_G, BIDIRECTIONAL_G, NUM_NOTES, TYPE_G
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
                       inp_dim=NUM_NOTES,
                       hid_dim=HIDDEN_DIM_G,
                       layers=LAYERS_G,
                       bidirectional=BIDIRECTIONAL_G).to(device)

        dense_input_features = (2 if BIDIRECTIONAL_G else 1) * HIDDEN_DIM_G
        self.dense = nn.Linear(in_features=dense_input_features, out_features=NUM_NOTES)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.dense(x)
        return F.softmax(x, dim=1)

    @staticmethod
    def noise(dims: Tuple):
        """
        Generates a 2-d vector of uniform sampled random values.
        :param dims: Tuple with the dimensions of the data.
        """
        return torch.randn(dims, dtype=torch.float).to(device)

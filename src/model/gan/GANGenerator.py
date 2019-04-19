from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from constants import HIDDEN_DIM_G, LAYERS_G, BIDIRECTIONAL_G, TYPE_G, SEQUENCE_LEN
from model.gan.RNN import RNN
from utils.tensors import device
from utils.typings import FloatTensor


class GANGenerator(nn.Module):
    def __init__(self, num_classes: int):
        """
        **Generator** network of a GAN model.
        Applies a RNN to a random noise generator, and passes each hidden state through a Dense Layer.
        Tries to generate realistic synthetic data to trick the **Discriminator** to make it say that it's real.
        """
        super(GANGenerator, self).__init__()

        self.rnn = RNN(architecture=TYPE_G,
                       inp_dim=1,
                       hid_dim=HIDDEN_DIM_G,
                       layers=LAYERS_G,
                       bidirectional=BIDIRECTIONAL_G).to(device)

        self.num_classes = num_classes
        self.dense_input_features = (2 if BIDIRECTIONAL_G else 1) * HIDDEN_DIM_G
        self.dense = nn.Linear(in_features=self.dense_input_features, out_features=num_classes)

    def forward(self, x_noise: FloatTensor, teacher_forcing=False, x_real=None):
        """
        Receives random noise as input and
        :param x_noise: x_0 (Noise data).
        :param teacher_forcing: Use real target outputs as each next input.
        :param x_real: Real data required if teacher_forcing == True.
        :return:
        """
        assert not teacher_forcing or x_real is not None, \
            'While using TeacherForcing, `x_real` must be feed with the real sequence data.'

        batch_size = x_noise.size(0)

        x = x_noise.view(batch_size, 1, 1)
        h = self.rnn.init_hidden(batch_size)  # h0
        c = self.rnn.init_hidden(batch_size)  # c0

        g_outputs = torch.zeros(batch_size, SEQUENCE_LEN, self.num_classes, device=device)
        d_inputs = torch.zeros(batch_size, SEQUENCE_LEN, 1, device=device)

        for i in range(SEQUENCE_LEN):
            x, (h, c) = self.rnn(x, (h, c))
            y = self.dense(x)
            y = F.softmax(y, dim=2)

            if teacher_forcing:
                x = x_real[:, i:i + 1, :]
            else:
                x = torch.argmax(y, dim=2, keepdim=True).to(torch.float)

            g_outputs[:, i:i + 1, :] = y
            d_inputs[:, i:i + 1] = x

        return g_outputs, d_inputs

    @staticmethod
    def noise(dims: Tuple):
        """
        Generates a 2-d vector of uniform sampled random values.
        :param dims: Tuple with the dimensions of the data.
        """
        return torch.randn(dims, device=device)

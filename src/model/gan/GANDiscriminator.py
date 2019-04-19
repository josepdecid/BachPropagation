import torch
from torch import nn
from torch.nn import functional as F

from constants import LAYERS_D, HIDDEN_DIM_D, BIDIRECTIONAL_D, TYPE_D, INPUT_FEATURES
from model.gan.RNN import RNN
from utils.tensors import device


class GANDiscriminator(nn.Module):
    def __init__(self):
        """
        Discriminator **D** network of a GAN model.
        Applies a RNN to the output of the Generator **G**, and passes each hidden state through a Dense Layer.
        Tries to generate discriminate between real and the synthetic (fake) data produced by **G**.
        """
        super(GANDiscriminator, self).__init__()

        self.rnn = RNN(architecture=TYPE_D,
                       inp_dim=1,
                       hid_dim=HIDDEN_DIM_D,
                       layers=LAYERS_D,
                       bidirectional=BIDIRECTIONAL_D).to(device)

        dense_input_features = (2 if BIDIRECTIONAL_D else 1) * HIDDEN_DIM_D
        self.dense_1 = nn.Linear(in_features=dense_input_features, out_features=1)

    def forward(self, x):
        batch_size = x.size(0)

        h = self.rnn.init_hidden(batch_size)  # h0
        c = self.rnn.init_hidden(batch_size)  # c0
        x, _ = self.rnn(x, (h, c))

        x = self.dense_1(x)
        x = torch.sigmoid(x)
        x = torch.mean(x, 1)
        x = x.view((-1,))

        return x

from torch import nn
from utils.constants import MAX_POLYPHONY

from model.RNN import RNN


class GANDiscriminator(nn.Module):
    def __init__(self, layers: int, hid_dim: int, bidirectional=False):
        """
        Discriminator **D** network of a GAN model.
        Applies a RNN to the output of the Generator **G**, and passes each hidden state through a Dense Layer.
        Tries to generate discriminate between real and the synthetic (fake) data produced by **G**.
        :param hid_dim: The number of features in the hidden state *h* of the RNN.
        """
        super(GANDiscriminator, self).__init__()

        self.rnn = RNN(architecture='GRU',
                       inp_dim=MAX_POLYPHONY,
                       hid_dim=hid_dim,
                       layers=layers,
                       bidirectional=bidirectional)

        dense_input_features = (2 if bidirectional else 1) * hid_dim
        self.dense = nn.Linear(in_features=dense_input_features,
                               out_features=1)

    def forward(self, x):
        x = self.rnn(x)
        x = self.dense(x)
        return x

from torch import nn

from model.utils.RNN import RNN


class GANGenerator(nn.Module):
    def __init__(self, hid_dim: int):
        """
        **Generator** network of a GAN model.
        Applies a RNN to a random noise generator, and passes each hidden state through a Dense Layer.
        Tries to generate realistic synthetic data to trick the **Discriminator** to make it say that it's real.
        :param hid_dim: The number of features in the hidden state *h* of the RNN.
        """
        super(GANGenerator, self).__init__()

        self.rnn = RNN(architecture='GRU', inp_dim=100, hid_dim=hid_dim, layers=2, bidirectional=False)
        self.dense = nn.Linear(in_features=hid_dim, out_features=hid_dim)

    def forward(self, x):
        x = self.rnn(x)
        x = self.dense(x)
        return x

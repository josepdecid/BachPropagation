from torch import nn

from model.RNN import RNN


class GANDiscriminator(nn.Module):
    def __init__(self, hid_dim: int):
        """
        Discriminator **D** network of a GAN model.
        Applies a RNN to the output of the Generator **G**, and passes each hidden state through a Dense Layer.
        Tries to generate discriminate between real and the synthetic (fake) data produced by **G**.
        :param hid_dim: The number of features in the hidden state *h* of the RNN.
        """
        super(GANDiscriminator, self).__init__()

        # TODO: Just to not explode
        self.rnn = RNN(architecture='GRU', inp_dim=100, hid_dim=hid_dim, layers=2, bidirectional=False)

    def forward(self, x):
        pass

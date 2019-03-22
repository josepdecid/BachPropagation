from torch import nn


class GANDiscriminator(nn.Module):
    def __init__(self, hid_dim: int):
        """
        Discriminator **D** network of a GAN model.
        Applies a RNN to the output of the Generator **G**, and passes each hidden state through a Dense Layer.
        Tries to generate discriminate between real and the synthetic (fake) data produced by **G**.
        :param hid_dim: The number of features in the hidden state *h* of the RNN.
        """
        super(GANDiscriminator, self).__init__()

    def forward(self, x):
        pass

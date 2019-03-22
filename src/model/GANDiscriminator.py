from torch import nn#, log, mean
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

        self.rnn = RNN(architecture='GRU', inp_dim=hid_dim, hid_dim=hid_dim, layers=2, bidirectional=True)
        self.dense = nn.Linear(in_features=hid_dim, out_features=1)

    def forward(self, x):
        x = self.rnn(x)
        x = self.dense(x)
        return x

    # def loss(self, fake, real):
    #     fake_out = self.forward(fake)
    #     real_out = self.forward(real)
    #
    #     loss = mean(-log(real_out) - log(1-fake_out))
    #     return loss

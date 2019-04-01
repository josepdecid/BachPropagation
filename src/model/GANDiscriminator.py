import torch
from torch import nn

from model.RNN import RNN
from utils.constants import LAYERS_D, HIDDEN_DIM_D, BIDIRECTIONAL_D, NUM_NOTES, TYPE_D
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
                       inp_dim=NUM_NOTES,
                       hid_dim=HIDDEN_DIM_D,
                       layers=LAYERS_D,
                       bidirectional=BIDIRECTIONAL_D).to(device)

        dense_input_features = (2 if BIDIRECTIONAL_D else 1) * HIDDEN_DIM_D
        self.dense = nn.Linear(in_features=dense_input_features, out_features=1)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.dense(x)
        x = torch.sigmoid(x)
        x = torch.mean(x, 1)
        x = x.view((-1,))
        return x

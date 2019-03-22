from torch import nn


class RNN(nn.Module):
    def __init__(self, architecture: str, inp_dim: int, hid_dim: int, layers=1, bidirectional=False):
        """
        Applies a multi-layer gated (*LSTM* or *GRU*) RNN to an input sequence.
        :param architecture: Gated RNN architecture (*LSTM* or *GRU*).
        :param inp_dim: The number of expected features in the input *x*.
        :param hid_dim: The number of features in the hidden state *h*.
        :param layers: Number of recurrent layers. Default: *1*.
        :param bidirectional: If `True`, becomes a bidirectional GRU. Default: `False`.
        """
        super(RNN, self).__init__()

        assert architecture in ('LSTM', 'GRU')
        gated_rnn = nn.LSTM if architecture == 'LSTM' else nn.GRU
        self.rnn = gated_rnn(input_size=inp_dim, hidden_size=hid_dim, num_layers=layers, bidirectional=bidirectional)

    def forward(self, x):
        return self.rnn(x)

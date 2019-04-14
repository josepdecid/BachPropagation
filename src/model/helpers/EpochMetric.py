from typing import Union

from constants import PLOT_COL


class EpochMetric:
    def __init__(self, epoch: int, g_loss: float, d_loss: Union[float, None]):
        self.epoch = epoch
        self.g_loss = g_loss
        self.d_loss = d_loss

    def print_metrics(self):
        if self.d_loss is None:
            print(f'(Pretrain) Generator loss: {self.g_loss:.6f}')
        else:
            print(f'Generator loss: {self.g_loss:.6f} | Discriminator loss: {self.d_loss:.6f}')

    def plot_loss(self, vis, plot='Training', title='Training Loss'):
        vis.plot_line(plot, 'Generator', title, 'Loss', [self.epoch], [self.g_loss], PLOT_COL['G'])
        if self.d_loss is not None:
            vis.plot_line(plot, 'Discriminator', None, None, [self.epoch], [self.d_loss], PLOT_COL['D'])

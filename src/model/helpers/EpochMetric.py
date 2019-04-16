from typing import Union, List
import matplotlib.pyplot as plt
import numpy as np

from constants import PLOT_COL
from model.helpers.VisdomPlotter import VisdomPlotter


class EpochMetric:
    def __init__(self, epoch: int, g_loss: float, d_loss: Union[float, None], cf: List[List[int]]):
        self.epoch = epoch
        self.g_loss = g_loss
        self.d_loss = d_loss
        self.cf = cf

    @property
    def tp_ratio(self):
        return self.cf[0][0] / (self.cf[0][0] + self.cf[0][1])

    @property
    def tn_ratio(self):
        return self.cf[1][1] / (self.cf[1][0] + self.cf[1][1])

    def print_metrics(self):
        if self.d_loss is None:
            print(f'(Pretrain) Generator loss: {self.g_loss:.6f}')
        else:
            print(f'Generator loss: {self.g_loss:.6f} | Discriminator loss: {self.d_loss:.6f}')
            print(f'Confusion matrix:'
                  f'\n\t{self.cf[0][0]:>6} {self.cf[0][1]:>6}'
                  f'\n\t{self.cf[1][0]:>6} {self.cf[1][1]:>6}')

    def plot_loss(self, vis, plot='Training', title='Training Loss'):
        vis.plot_line(plot, 'Generator', title, 'Loss', [self.epoch], [self.g_loss], PLOT_COL['G'])
        if self.d_loss is not None:
            vis.plot_line(plot, 'Discriminator', None, None, [self.epoch], [self.d_loss], PLOT_COL['D'])

    def plot_confusion_matrix(self, vis: VisdomPlotter):
        # True positives
        vis.plot_line(plot_name='ConfusionMatrix',
                      line_label='True Positives',
                      title='Confusion Matrix Ratios',
                      y_label='Ratio',
                      x=[self.epoch],
                      y=[self.tp_ratio])

        # True Negative Ratios
        vis.plot_line(plot_name='ConfusionMatrix',
                      line_label='True Negatives',
                      x=[self.epoch],
                      y=[self.tn_ratio])

        # Confusion Matrix
        cf = np.array(self.cf)
        fig, ax = plt.subplots()
        ax.imshow(cf, cmap=plt.cm.Oranges)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Predicted Real', 'Predicted fake'])
        ax.set_yticklabels(['Actual Real', 'Actual fake'])
        ax.text(0, 0, f'TP {self.cf[0][0]}', ha='center', va='center', color='black')
        ax.text(1, 0, f'FP {self.cf[0][1]}', ha='center', va='center', color='black')
        ax.text(0, 1, f'FN {self.cf[1][0]}', ha='center', va='center', color='black')
        ax.text(1, 1, f'TN {self.cf[1][1]}', ha='center', va='center', color='black')
        ax.set_title(f'Confusion Matrix (Epoch {self.epoch})')
        fig.tight_layout()
        vis.display_matplot_figure(fig, plot_name='CF')

import logging
from datetime import datetime
from typing import List

import torch
from tqdm import tqdm
from visdom import Visdom

from dataset.MusicDataset import MusicDataset
from dataset.preprocessing.reconstructor import reconstruct_midi
from model.GANGenerator import GANGenerator
from model.GANModel import GANModel
from constants import EPOCHS, NUM_NOTES, CKPT_STEPS, CHECKPOINTS_PATH, SAMPLE_STEPS, FLAGS, PLOT_COL
from utils.tensors import device


class VisdomLinePlotter:
    def __init__(self):
        self.viz = Visdom(env='BachPropagation')
        self.plots = {}

    def plot_line(self, plot, line, title, y_label, x, y, color=None):
        if plot not in self.plots:
            opts = {'title': title, 'xlabel': 'Epochs', 'ylabel': y_label, 'linecolor': color}
            if line is not None:
                opts['legend'] = [line]
            self.plots[plot] = self.viz.line(X=x, Y=y, opts=opts)
        else:
            self.viz.line(X=x, Y=y, win=self.plots[plot], name=line, update='append', opts={'linecolor': color})

    def add_song(self, path):
        self.viz.audio(audiofile=path, tensor=None)


class EpochMetric:
    def __init__(self, epoch: int, g_loss: float, d_loss: float):
        self.epoch = epoch
        self.g_loss = g_loss
        self.d_loss = d_loss

    def print_metrics(self):
        print(f'Generator loss: {self.g_loss:.6f} | Discriminator loss: {self.d_loss:.6f}')

    def plot_loss(self, vis):
        vis.plot_line('Loss', 'Generator', f'Model Loss', 'Loss', [self.epoch], [self.g_loss], PLOT_COL['G'])
        vis.plot_line('Loss', 'Discriminator', None, None, [self.epoch], [self.d_loss], PLOT_COL['D'])


class Trainer:
    def __init__(self, model: GANModel, dataset: MusicDataset):
        self.model = model

        self.data = dataset
        self.loader = dataset.get_dataloader(shuffle=True)

        # Visualization
        if FLAGS['viz']:
            self.vis = VisdomLinePlotter()

        # Accuracies and Losses
        self.metrics: List[EpochMetric] = []

    def train(self):
        logging.info(f'Training the model...')

        for epoch in range(1, EPOCHS + 1):
            metric = self._train_epoch(epoch)

            if FLAGS['viz']:
                metric.plot_loss(self.vis)
                self.vis.plot_line('LR_G', None, 'LR Generator', 'LR',
                                   [epoch], [self.model.g_optimizer.param_groups[0]['lr']], PLOT_COL['G'])
                self.vis.plot_line('LR_D', None, 'LR Discriminator', 'LR',
                                   [epoch], [self.model.d_optimizer.param_groups[0]['lr']], PLOT_COL['D'])
            metric.print_metrics()

            if epoch % SAMPLE_STEPS == 0:
                sample = self.generate_sample(length=500)
                reconstruct_midi(title=f'Sample {epoch}', data=sample)
                # TODO: Visdom doesn't accept MIDI files.
                #  Should convert to WAV or find an alternative for visualization.
                # if FLAGS['viz']:
                #     self.vis.add_song(midi_file_path)

            if epoch % CKPT_STEPS == 0:
                ts = str(datetime.now()).split('.')[0].replace(' ', '_')
                torch.save(self.model.generator.state_dict(), f'{CHECKPOINTS_PATH}/{ts}_G{epoch}.pt')
                torch.save(self.model.discriminator.state_dict(), f'{CHECKPOINTS_PATH}/{ts}_D{epoch}.pt')

    def generate_sample(self, length: int):
        self.model.eval_mode()
        with torch.no_grad():
            noise_data = GANGenerator.noise((1, length, NUM_NOTES))
            sample_data = self.model.generator(noise_data)
            sample_notes = sample_data.argmax(2)
            return sample_notes.view(-1).cpu().numpy()

    def _train_epoch(self, epoch: int) -> EpochMetric:
        """
        Trains the model for one epoch with the given training data and using the specified optimizer and loss function.
        :param epoch: Current epoch index.
        :return: Accuracy and number of correct predictions for training data in the current epoch.
        """
        sum_loss_g = []
        sum_loss_d = []

        batch_data = enumerate(tqdm(self.loader, desc=f'Epoch {epoch}: ', ncols=100))
        for batch_idx, features in batch_data:
            features = features.requires_grad_().to(device)

            # if current_loss_d >= 0.7 * current_loss_g:
            d_loss = self._train_discriminator(features)
            # current_loss_d = d_loss
            sum_loss_d.append(d_loss)

            # if current_loss_g >= 0.7 * current_loss_d:
            g_loss = self._train_generator(data=features)
            # current_loss_g = g_loss
            sum_loss_g.append(g_loss)

        g_loss = sum(sum_loss_g) / len(sum_loss_g)
        d_loss = sum(sum_loss_d) / len(sum_loss_d)

        self.model.g_scheduler.step(metrics=g_loss)
        self.model.d_scheduler.step(metrics=d_loss)

        return EpochMetric(epoch, g_loss, d_loss)

    def _train_generator(self, data):
        logging.debug('Training Generator')

        batch_size = data.size(0)
        time_steps = data.size(1)

        noise_data = GANGenerator.noise((batch_size, time_steps, NUM_NOTES))
        fake_data = self.model.generator(noise_data)

        # Reset gradients
        self.model.g_optimizer.zero_grad()

        # Forwards pass to get logits
        prediction = self.model.discriminator(fake_data)

        # Calculate gradients w.r.t parameters and backpropagate
        loss = -self.model.g_criterion(d_g_z=prediction)
        loss.backward()

        # Update parameters
        self.model.g_optimizer.step()

        return -(loss.item())

    def _train_discriminator(self, real_data) -> float:
        logging.debug('Training Discriminator')

        batch_size = real_data.size(0)
        time_steps = real_data.size(1)

        # Reset gradients
        self.model.d_optimizer.zero_grad()

        # Train on real data
        real_predictions = self.model.discriminator(real_data)

        # Train on fake data
        noise_data = GANGenerator.noise((batch_size, time_steps, NUM_NOTES))
        fake_data = self.model.generator(noise_data).detach()
        fake_predictions = self.model.discriminator(fake_data)

        # Calculate loss and optimize
        loss = -self.model.d_criterion(d_x=real_predictions, d_g_z=fake_predictions)
        loss.backward()

        # Update parameters
        self.model.d_optimizer.step()

        return -(loss.item())

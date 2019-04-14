import logging
from datetime import datetime
from typing import List

import torch
from tqdm import tqdm

from constants import EPOCHS, NUM_NOTES, CKPT_STEPS, CHECKPOINTS_PATH, \
    SAMPLE_STEPS, FLAGS, PLOT_COL, PRETRAIN_G, PRETRAIN_D
from dataset.MusicDataset import MusicDataset
from dataset.preprocessing.reconstructor import reconstruct_midi
from model.gan.GANGenerator import GANGenerator
from model.gan.GANModel import GANModel
from model.helpers.EpochMetric import EpochMetric
from model.helpers.VisdomPlotter import VisdomPlotter
from utils.tensors import device, zeros_target, ones_target
from utils.typings import FloatTensor, NDArray


class Trainer:
    def __init__(self, model: GANModel, dataset: MusicDataset):
        self.model = model

        self.data = dataset
        self.loader = dataset.get_dataloader(shuffle=True)

        # Visualization
        if FLAGS['viz']:
            self.vis = VisdomPlotter()

        # Accuracies and Losses
        self.metrics: List[EpochMetric] = []

    def train(self):
        """
        Train the GAN model with some (optional) pretraining.
        Store model checkpoints every constants.CKPT_STEPS epochs.
        Generate sample songs every constants.SAMPLE_STEPS epochs.
        """
        logging.info(f'Training the model...')
        logging.info(f'Generator model {self.model.generator}')
        logging.info(f'Discriminator model {self.model.discriminator}')

        # PRE-TRAINING
        epochs_pretraining = max(PRETRAIN_G, PRETRAIN_D)
        for epoch in range(1, epochs_pretraining + 1):
            metric = self._pretrain_epoch(epoch)

            if FLAGS['viz']:
                metric.plot_loss(self.vis, plot='Pretraining', title='Pretraining Loss')
            metric.print_metrics()

        # TRAINING
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
                reconstruct_midi(title=f'Sample_{epoch}', data=sample)
                # TODO: Visdom doesn't accept MIDI files.
                #  Should convert to WAV or find an alternative for visualization.
                # if FLAGS['viz']:
                #     self.vis.add_song(midi_file_path)

            if epoch % CKPT_STEPS == 0:
                ts = str(datetime.now()).split('.')[0].replace(' ', '_')
                torch.save(self.model.generator.state_dict(), f'{CHECKPOINTS_PATH}/{ts}_G{epoch}.pt')
                torch.save(self.model.discriminator.state_dict(), f'{CHECKPOINTS_PATH}/{ts}_D{epoch}.pt')

    def generate_sample(self, length: int) -> NDArray:
        """
        Generate synthetic song from the Generator.
        :param length: Length of the song.
        :return: NDArray with song data
        """
        self.model.eval_mode()
        with torch.no_grad():
            noise_data = GANGenerator.noise((1, length, NUM_NOTES))
            sample_data = self.model.generator(noise_data)
            sample_notes = sample_data.argmax(2)
            return sample_notes.view(-1).cpu().numpy()

    def _pretrain_epoch(self, epoch: int) -> EpochMetric:
        """
        Pretrain the model for one epoch trying to obtain the real data from the Generator.
        :param epoch: Current epoch index.
        :return: Epoch performance metrics.
        """
        self.model.train_mode()

        sum_loss_g = 0
        sum_loss_d = 0

        batch_data = enumerate(tqdm(self.loader, desc=f'Epoch {epoch}: ', ncols=100))
        for batch_idx, features in batch_data:
            features = features.to(device)
            batch_size = features.size(0)

            g_loss = self._train_generator(real_data=features, pretraining=True)
            sum_loss_g += g_loss * batch_size

            d_loss = self._train_discriminator(real_data=features, pretraining=True)
            sum_loss_d += d_loss * batch_size

        g_loss = sum_loss_g / len(self.loader.dataset)
        d_loss = sum_loss_d / len(self.loader.dataset)
        return EpochMetric(epoch, g_loss, d_loss)

    def _train_epoch(self, epoch: int) -> EpochMetric:
        """
        Train the model for one epoch with the classical GAN training approach.
        :param epoch: Current epoch index.
        :return: Epoch performance metrics.
        """
        self.model.train_mode()

        sum_loss_g = 0
        sum_loss_d = 0

        batch_data = enumerate(tqdm(self.loader, desc=f'Epoch {epoch}: ', ncols=100))
        for batch_idx, features in batch_data:
            features = features.to(device)
            batch_size = features.size(0)

            # if current_loss_d >= 0.7 * current_loss_g:
            d_loss = self._train_discriminator(features)
            # current_loss_d = d_loss
            sum_loss_d += d_loss * batch_size

            # if current_loss_g >= 0.7 * current_loss_d:
            g_loss = self._train_generator(real_data=features)
            # current_loss_g = g_loss
            sum_loss_g += g_loss * batch_size

        g_loss = sum_loss_g / len(self.loader.dataset)
        d_loss = sum_loss_d / len(self.loader.dataset)

        self.model.g_scheduler.step(metrics=g_loss)
        self.model.d_scheduler.step(metrics=d_loss)

        return EpochMetric(epoch, g_loss, d_loss)

    def _train_generator(self, real_data, pretraining=False) -> float:
        """
        Train GAN Generator performing an optimizer step.
        :param real_data: Real inputs from the dataset
        :param pretraining: Boolean indicating whether it's pretraining or training.
        :return: Discriminator loss from the specified criterion.
        """
        logging.debug('Pretraining Generator') if pretraining else logging.debug('Training Generator')

        batch_size = real_data.size(0)
        time_steps = real_data.size(1)

        noise_data = GANGenerator.noise((batch_size, time_steps, NUM_NOTES))
        fake_data = self.model.generator(noise_data)

        # Reset gradients
        self.model.g_optimizer.zero_grad()

        # Forwards pass to get logits
        # Calculate gradients w.r.t parameters and backpropagate
        if pretraining:
            loss = self.model.pretraining_criterion(fake_data.view(batch_size, NUM_NOTES, -1),
                                                    real_data.argmax(dim=2).long())
        else:
            prediction = self.model.discriminator(fake_data)
            loss = self.model.training_criterion(prediction, ones_target((batch_size,)))

        loss.backward()

        # Update parameters
        self.model.g_optimizer.step()

        return loss.item()

    def _train_discriminator(self, real_data: FloatTensor, pretraining=False) -> float:
        """
        Train GAN Discriminator performing an optimizer step.
        :param real_data: Real inputs from the dataset
        :return: Discriminator loss from the specified criterion.
        """
        logging.debug('Training Discriminator')

        batch_size = real_data.size(0)
        time_steps = real_data.size(1)

        # Reset gradients
        self.model.d_optimizer.zero_grad()

        #Predictions on real data
        real_predictions = self.model.discriminator(real_data)

        if pretraining:
            real_loss = self.model.pretraining_discriminator_criterion(real_predictions, torch.ones(real_predictions.shape))
            real_loss.backward()
            fake_loss = 0.

        else:
            # Train on real data
            real_loss = self.model.training_criterion(real_predictions, ones_target((batch_size,)))
            real_loss.backward()

            # Train on fake data
            noise_data = GANGenerator.noise((batch_size, time_steps, NUM_NOTES))
            fake_data = self.model.generator(noise_data).detach()
            fake_predictions = self.model.discriminator(fake_data)
            fake_loss = self.model.training_criterion(fake_predictions, zeros_target((batch_size,)))
            fake_loss.backward()

            # Update parameters
            self.model.d_optimizer.step()

        return (real_loss + fake_loss).item()

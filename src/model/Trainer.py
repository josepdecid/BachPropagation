import logging
from datetime import datetime
from typing import List, Tuple

import torch
from tqdm import tqdm

from constants import EPOCHS, CKPT_STEPS, CHECKPOINTS_PATH, SAMPLE_STEPS, FLAGS, PLOT_COL, PRETRAIN_G, PRETRAIN_D, \
    T_LOSS_BALANCER, SEQUENCE_LEN
from dataset.MusicDataset import MusicDataset
from dataset.preprocessing.reconstructor import reconstruct_midi
from model.gan.GANGenerator import GANGenerator
from model.gan.GANModel import GANModel
from model.helpers.EpochMetric import EpochMetric
from model.helpers.VisdomPlotter import VisdomPlotter
from utils.tensors import device, zeros_target, ones_target
from utils.typings import FloatTensor


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

        # sample = generate_sample(model=self.model, length=500)
        # reconstruct_midi(title=f'Sample_test', raw_data=sample)

        current_loss_d = 1e99
        current_loss_g = 1e99

        # TRAINING
        for epoch in range(1, EPOCHS + 1):
            metric = self._train_epoch(epoch, current_loss_d, current_loss_g)

            if FLAGS['viz']:
                metric.plot_loss(self.vis)
                metric.plot_confusion_matrix(self.vis)
                self.vis.plot_line('LR_G', None, 'LR Generator', 'LR',
                                   [epoch], [self.model.g_optimizer.param_groups[0]['lr']], PLOT_COL['G'])
                self.vis.plot_line('LR_D', None, 'LR Discriminator', 'LR',
                                   [epoch], [self.model.d_optimizer.param_groups[0]['lr']], PLOT_COL['D'])
            metric.print_metrics()

            if epoch % SAMPLE_STEPS == 0:
                sample = self.generate_sample(length=5000)
                reconstruct_midi(title=f'Sample_{epoch}', raw_data=sample)
                # TODO: Visdom doesn't accept MIDI files.
                #  Should convert to WAV or find an alternative for visualization.
                # if FLAGS['viz']:
                #     self.vis.add_song(midi_file_path)

            if epoch % CKPT_STEPS == 0:
                ts = str(datetime.now()).split('.')[0].replace(' ', '_')
                torch.save(self.model.generator.state_dict(), f'{CHECKPOINTS_PATH}/{ts}_G{epoch}.pt')
                torch.save(self.model.discriminator.state_dict(), f'{CHECKPOINTS_PATH}/{ts}_D{epoch}.pt')

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
        for batch_idx, (features, labels) in batch_data:
            features = features.to(device)
            labels = labels.to(device)
            batch_size = features.size(0)

            if epoch < PRETRAIN_G:
                g_loss = self._train_generator(real_data=features, pretraining_labels=labels)
                sum_loss_g += g_loss * batch_size

            # if epoch < PRETRAIN_D:
            #    d_loss = self._train_discriminator(real_data=features)
            #    sum_loss_d += d_loss * batch_size

        g_loss = sum_loss_g / len(self.loader.dataset)
        d_loss = sum_loss_d / len(self.loader.dataset)

        self.model.g_scheduler.step(metrics=g_loss)

        return EpochMetric(epoch, g_loss, d_loss, None)

    def _train_epoch(self, epoch: int, current_loss_d=0.0, current_loss_g=0.0) -> EpochMetric:
        """
        Train the model for one epoch with the classical GAN training approach.
        :param epoch: Current epoch index.
        :return: Epoch performance metrics.
        """
        self.model.train_mode()

        sum_loss_g, sum_loss_d = 0, 0
        sum_t_pos, sum_t_neg = 0, 0

        batch_data = enumerate(tqdm(self.loader, desc=f'Epoch {epoch}: ', ncols=100))
        for batch_idx, (features, _) in batch_data:
            features = features.to(device)
            batch_size = features.size(0)

            # if current_loss_d >= T_LOSS_BALANCER * current_loss_g:
            #     d_loss, t_pos, t_neg = self._train_discriminator(real_data=features)
            #     current_loss_d = d_loss
            #     sum_loss_d += d_loss * batch_size
            #     sum_t_pos += t_pos
            #     sum_t_neg += t_neg
            # else:
            #     logging.debug('Freezing Discriminator')

            if current_loss_g >= T_LOSS_BALANCER * current_loss_d:
                g_loss = self._train_generator(real_data=features)
                current_loss_g = g_loss
                sum_loss_g += g_loss * batch_size
            else:
                logging.debug('Freezing Generator')

        len_data = len(self.loader.dataset)
        sum_loss_g /= len_data
        sum_loss_d /= len_data
        self.model.g_scheduler.step(metrics=sum_loss_g)
        self.model.d_scheduler.step(metrics=sum_loss_d)

        confusion_matrix = [[sum_t_pos, len_data - sum_t_pos],
                            [len_data - sum_t_neg, sum_t_neg]]

        return EpochMetric(epoch, sum_loss_g, sum_loss_d, confusion_matrix)

    def _train_generator(self, real_data, pretraining_labels=None) -> float:
        """
        Train GAN Generator performing an optimizer step.
        :param real_data: Real inputs from the dataset
        :param pretraining_labels: Boolean indicating whether it's pretraining or training.
        :return: Discriminator loss from the specified criterion.
        """
        pretraining = pretraining_labels is not None
        if pretraining:
            logging.debug('Training Generator')
        else:
            logging.debug('Pretraining Generator')

        batch_size = real_data.size(0)
        noise_data = GANGenerator.noise((batch_size, 1))

        # Reset gradients
        self.model.g_optimizer.zero_grad()

        # Forwards pass to get logits
        # Calculate gradients w.r.t parameters and backpropagate
        if pretraining:
            predictions = self.model.generator(noise_data, teacher_forcing=True, x_real=real_data)
            loss = self.model.pretrain_criterion(predictions.view(batch_size * SEQUENCE_LEN, -1),
                                                 real_data.view(batch_size * SEQUENCE_LEN).to(torch.long))
        else:
            g_predictions = self.model.generator(noise_data)
            d_predictions = self.model.discriminator(g_predictions)
            loss = self.model.train_criterion(d_predictions, ones_target((batch_size,)))

        loss.backward()

        # Update parameters
        self.model.g_optimizer.step()

        return loss.item()

    def _train_discriminator(self, real_data: FloatTensor) -> Tuple[float, int, int]:
        """
        Train GAN Discriminator performing an optimizer step.
        :param real_data: Real inputs from the dataset
        :return: Discriminator loss from the specified criterion.
        """
        logging.debug('Training Discriminator')

        batch_size = real_data.size(0)
        time_steps = real_data.size(1)

        zeros = zeros_target((batch_size,))
        ones = ones_target((batch_size,))

        # Reset gradients
        self.model.d_optimizer.zero_grad()

        # Predictions on real data
        real_predictions = self.model.discriminator(real_data)

        # Train on real data
        real_loss = self.model.pretrain_criterion(real_predictions, ones)
        real_loss.backward()

        # Train on fake data
        noise_data = GANGenerator.noise((batch_size, time_steps))
        fake_data = self.model.generator(noise_data).detach()
        fake_predictions = self.model.discriminator(fake_data)
        fake_loss = self.model.pretrain_criterion(fake_predictions, zeros)
        fake_loss.backward()

        # Update parameters
        self.model.d_optimizer.step()

        true_positives = (real_predictions > 0.5).to(torch.float).eq(ones).sum().item()
        true_negatives = (fake_predictions > 0.5).to(torch.float).eq(zeros).sum().item()

        return (real_loss + fake_loss).item(), true_positives, true_negatives

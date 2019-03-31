import logging

import sys
from typing import Tuple

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset.preprocessing.reconstructor import reconstruct_midi
from dataset.MusicDataset import MusicDataset
from model.GANGenerator import GANGenerator
from model.GANModel import GANModel
from utils.constants import BATCH_SIZE, EPOCHS, NUM_NOTES
from utils.tensors import device


def generate_sample(model: GANModel, time_steps: int):
    noise_data = GANGenerator.noise((BATCH_SIZE, time_steps, NUM_NOTES))
    sample_data = model.generator(noise_data)
    _, sample_notes = sample_data.max(0)

    return sample_notes


def train_generator(model: GANModel, data):
    logging.debug('Training Generator')

    time_steps = data.size(1)
    noise_data = GANGenerator.noise((BATCH_SIZE, time_steps, NUM_NOTES))
    fake_data = model.generator(noise_data)

    # Reset gradients
    model.g_optimizer.zero_grad()

    # Forwards pass to get logits
    prediction = model.discriminator(fake_data)

    # Calculate gradients w.r.t parameters and backpropagate
    loss = model.g_criterion(prediction, Variable(torch.ones(BATCH_SIZE, 1)))
    loss.backward()

    # Update parameters
    model.g_optimizer.step()
    return loss


def train_discriminator(model: GANModel, data):
    logging.debug('Training Discriminator')

    def train_data(x, fake: bool):
        # Forwards pass to get logits
        prediction = model.discriminator(x)

        # Calculate gradients w.r.t parameters and backpropagate
        n = real_data.size(0)
        target = Variable(torch.zeros(n)) if fake else Variable(torch.ones(n))

        loss = model.d_criterion(prediction, target)
        loss.backward()

        return loss

    time_steps = data.size(1)

    real_data = Variable(data)

    noise_data = GANGenerator.noise((BATCH_SIZE, time_steps, NUM_NOTES))
    fake_data = model.generator(noise_data).detach()

    # Reset gradients
    model.d_optimizer.zero_grad()

    # Train on real data
    loss_real = train_data(real_data, fake=False)

    # Train on fake data
    loss_fake = train_data(fake_data, fake=True)

    # Update parameters
    model.d_optimizer.step()
    return loss_real + loss_fake


def train_epoch(model: GANModel, loader: DataLoader) -> Tuple[float, float]:
    current_loss_g = 1.e10
    current_loss_d = 1.e10

    sum_loss_g = []
    sum_loss_d = []

    for batch_idx, batch_data in enumerate(loader):
        logging.debug(f'Batch {batch_idx}/{len(loader)}')

        if current_loss_d >= 0.7 * current_loss_g:
            d_loss = train_discriminator(model=model, data=batch_data)
            current_loss_d = d_loss
            sum_loss_d.append(d_loss)

        if current_loss_g >= 0.7 * current_loss_d:
            g_loss = train_generator(model=model, data=batch_data)
            current_loss_g = g_loss
            sum_loss_g.append(g_loss)

    return sum(sum_loss_g) / len(sum_loss_g), sum(sum_loss_d) / len(sum_loss_d)


def train(model: GANModel, dataset: MusicDataset):
    logging.info(f'Training the model...')
    for epoch in range(EPOCHS):
        g_loss, d_loss = train_epoch(model=model, loader=dataset.get_dataloader(shuffle=True))

        print(f'Epoch {epoch:4} | Generator loss: {g_loss:.6f} ; Discriminator loss: {d_loss:.6f}')

        sample = generate_sample(model, 30000)[:, 0].numpy()
        reconstruct_midi(title=f'Sample {epoch}', data=sample)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info(f'Running Python {sys.version.split()[0]} with PyTorch {torch.__version__} in {device}')

    train(model=GANModel(),
          dataset=MusicDataset())

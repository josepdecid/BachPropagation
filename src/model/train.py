import logging

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset.MusicDataset import MusicDataset
from model.GANGenerator import GANGenerator
from model.GANModel import GANModel
from utils.constants import BATCH_SIZE, MAX_POLYPHONY, EPOCHS, NUM_NOTES


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


def train_epoch(model: GANModel, loader: DataLoader) -> float:
    for features in loader:
        d_loss = train_discriminator(model=model, data=features)
        g_loss = train_generator(model=model, data=features)
        return d_loss + g_loss


def train(model: GANModel, dataset: MusicDataset):
    logging.info(f'Training the model...')
    for epoch in range(EPOCHS):
        loss = train_epoch(model=model, loader=dataset.get_dataloader(BATCH_SIZE, shuffle=True))
        print(f'Epoch {epoch}: Training loss = {loss}')
        sample = generate_sample(model, dataset.longest_song)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    train(model=GANModel(),
          dataset=MusicDataset())

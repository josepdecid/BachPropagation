import logging
import sys
from datetime import datetime
from typing import Tuple

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset.MusicDataset import MusicDataset
from dataset.preprocessing.reconstructor import reconstruct_midi
from model.GANGenerator import GANGenerator
from model.GANModel import GANModel
from utils.constants import EPOCHS, NUM_NOTES, CKPT_STEPS, CHECKPOINTS_PATH, SAMPLE_STEPS
from utils.tensors import device


def generate_sample(model: GANModel, time_steps: int):
    model.eval_mode()
    with torch.no_grad():
        noise_data = GANGenerator.noise((1, time_steps, NUM_NOTES))
        sample_data = model.generator(noise_data)
        sample_notes = sample_data.argmax(2)

        return sample_notes.view(-1)


def train_generator(model: GANModel, data):
    logging.debug('Training Generator')

    batch_size = data.size(0)
    time_steps = data.size(1)

    noise_data = GANGenerator.noise((batch_size, time_steps, NUM_NOTES))
    fake_data = model.generator(noise_data)

    # Reset gradients
    model.g_optimizer.zero_grad()

    # Forwards pass to get logits
    prediction = model.discriminator(fake_data)

    # Calculate gradients w.r.t parameters and backpropagate
    loss = model.g_criterion(prediction)
    loss.backward()

    # Update parameters
    model.g_optimizer.step()
    return loss


def train_discriminator(model: GANModel, data):
    logging.debug('Training Discriminator')

    batch_size = data.size(0)
    time_steps = data.size(1)

    # Reset gradients
    model.d_optimizer.zero_grad()

    # Train on real data
    real_data = Variable(data)
    real_predictions = model.discriminator(real_data)

    # Train on fake data
    noise_data = GANGenerator.noise((batch_size, time_steps, NUM_NOTES))
    fake_data = model.generator(noise_data).detach()
    fake_predictions = model.discriminator(fake_data)

    # Calculate loss and optimize
    loss = model.d_criterion(real_predictions, fake_predictions)
    loss.backward()

    # Update parameters
    model.d_optimizer.step()

    return loss


def train_epoch(model: GANModel, loader: DataLoader) -> Tuple[float, float]:
    current_loss_g = 1.e10
    current_loss_d = 1.e10

    sum_loss_g = []
    sum_loss_d = []

    model.train_mode()
    for batch_idx, batch_data in enumerate(loader):
        logging.debug(f'Batch {batch_idx}/{len(loader)}')
        batch_data = batch_data.to(device)

        # if current_loss_d >= 0.7 * current_loss_g:
        d_loss = train_discriminator(model=model, data=batch_data)
        current_loss_d = d_loss
        sum_loss_d.append(d_loss)

        # if current_loss_g >= 0.7 * current_loss_d:
        g_loss = train_generator(model=model, data=batch_data)
        current_loss_g = g_loss
        sum_loss_g.append(g_loss)

    return sum(sum_loss_g) / len(sum_loss_g), sum(sum_loss_d) / len(sum_loss_d)


def train(model: GANModel, dataset: MusicDataset):
    logging.info(f'Training the model...')
    for epoch in range(1, EPOCHS + 1):
        g_loss, d_loss = train_epoch(model=model, loader=dataset.get_dataloader(shuffle=True))
        print(f'Epoch {epoch:4} | Generator loss: {g_loss:.6f} ; Discriminator loss: {d_loss:.6f}')

        if epoch % SAMPLE_STEPS == 0:
            sample = generate_sample(model, dataset.longest_song).cpu().numpy()
            reconstruct_midi(title=f'Sample {epoch}', data=sample)

        if epoch % CKPT_STEPS == 0:
            ts = str(datetime.now()).split('.')[0].replace(' ', '_')
            torch.save(model.generator.state_dict(), f'{CHECKPOINTS_PATH}/{ts}_G{epoch}.pt')
            torch.save(model.discriminator.state_dict(), f'{CHECKPOINTS_PATH}/{ts}_D{epoch}.pt')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info(f'Running Python {sys.version.split()[0]} with PyTorch {torch.__version__} in {device}')

    train(model=GANModel(),
          dataset=MusicDataset())

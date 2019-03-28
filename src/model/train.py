import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset.MusicDataset import MusicDataset
from model.GANGenerator import GANGenerator
from model.GANModel import GANModel
from utils.constants import BATCH_SIZE, MAX_POLYPHONY
from utils.typings import Optimizer, Criterion, NNet


def train_generator(optimizer: Optimizer, criterion: Criterion, discriminator: NNet, fake_data):
    # Reset gradients
    optimizer.zero_grad()

    # Forwards pass to get logits
    prediction = discriminator(fake_data)

    # Calculate gradients w.r.t parameters and backpropagate
    loss = criterion(prediction, Variable(torch.ones(fake_data.size(0), 1)))
    loss.backward()

    # Update parameters
    optimizer.step()
    return loss


def train_discriminator(optimizer: Optimizer, criterion: Criterion, discriminator: NNet, real_data, fake_data):
    def train_data(data, fake: bool):
        # Forwards pass to get logits
        prediction = discriminator(data)

        # Calculate gradients w.r.t parameters and backpropagate
        n = real_data.size(0)
        target = Variable(torch.zeros(n)) if fake else Variable(torch.ones(n))
        loss = criterion(prediction, target)
        loss.backward()

        return loss

    # Reset gradients
    optimizer.zero_grad()

    # Train on real data
    loss_real = train_data(real_data, fake=False)

    # Train on fake data
    loss_fake = train_data(fake_data, fake=True)

    # Update parameters
    optimizer.step()
    return loss_real + loss_fake


def train_epoch(model: GANModel, loader: DataLoader) -> float:
    model.train_mode()
    for batch in loader:
        batch_size = batch.size(0)
        input_dim = batch.size(1)

        # Train Discriminator
        real_data = Variable(batch)
        real_data = real_data.view(input_dim, batch_size, MAX_POLYPHONY)

        noise_data = GANGenerator.noise(batch_size, input_dim)
        noise_data = noise_data.view(input_dim, batch_size, MAX_POLYPHONY)
        fake_data = model.generator(noise_data).detach()

        d_loss = train_discriminator(optimizer=model.d_optimizer,
                                     criterion=model.d_criterion,
                                     discriminator=model.discriminator,
                                     real_data=real_data,
                                     fake_data=fake_data)

        # Train Generator
        noise_data = GANGenerator.noise(batch_size, input_dim)
        noise_data = noise_data.view(input_dim, batch_size, MAX_POLYPHONY)
        fake_data = model.generator(noise_data)

        g_loss = train_generator(optimizer=model.g_optimizer,
                                 criterion=model.g_criterion,
                                 discriminator=model.discriminator,
                                 fake_data=fake_data)

        return d_loss + g_loss


def train(model: GANModel, epochs: int, train_loader: DataLoader):
    for epoch in range(epochs):
        loss = train_epoch(model=model, loader=train_loader)
        print(f'Epoch {epoch}: Training loss = {loss}')


if __name__ == '__main__':
    def main():
        model = GANModel()
        train_songs = MusicDataset()

        train(model=model,
              epochs=100,
              train_loader=train_songs.get_dataloader(BATCH_SIZE))


    main()

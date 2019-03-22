import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset.MusicDataset import MusicDataset
from model.GANModel import GANModel
from utils.constants import BATCH_SIZE


def train_epoch():
    pass


def train(model: GANModel, epochs: int, train_loader: DataLoader, test_loader: DataLoader):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        loss = train_epoch()


if __name__ == '__main__':
    model = GANModel()
    train_songs = MusicDataset()
    test_songs = MusicDataset()

    train(model=model,
          epochs=100,
          train_loader=train_songs.get_dataloader(BATCH_SIZE),
          test_loader=test_songs.get_dataloader(BATCH_SIZE))

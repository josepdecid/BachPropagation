import glob
import logging
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader

from constants import DATASET_PATH, BATCH_SIZE, INPUT_FEATURES
from utils.tensors import use_cuda
from utils.typings import File, FloatTensor


class MusicDataset(Dataset):
    """Music dataset representation, which reads, parses, adds padding and generates DataLoader for training"""

    def __init__(self):
        logging.info('Loading music data...')

        self.songs = []
        self.longest_song = 0
        for path in glob.glob(f'{DATASET_PATH}/*.txt'):
            with open(path, mode='r') as f:
                self.songs.append(self._read_song(f))

        self.padded_songs = self._apply_padding()

    def __getitem__(self, index: int) -> FloatTensor:
        return self.padded_songs[index]

    def __len__(self) -> int:
        return len(self.songs)

    def get_dataloader(self, shuffle=False) -> DataLoader:
        """
        Builds a data loader object with the padded songs in the dataset.
        :param shuffle: Randomize the order of the songs each time that it's called (each epoch).
        :return: PyTorch DataLoader object.
        """
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        return DataLoader(self, batch_size=BATCH_SIZE, shuffle=shuffle, **kwargs)

    def _apply_padding(self) -> List[FloatTensor]:
        """
        Applies padding to songs to be all of the size of the longest one.
        :return: Padded songs with Null-One hot encoding representations for padded data.
        """
        padded_songs = []
        for song in self.songs:
            padded_song = torch.zeros((self.longest_song, INPUT_FEATURES), dtype=torch.float)
            padded_song[:len(song), :] = torch.tensor(song)
            padded_songs.append(padded_song)
        return padded_songs

    def _update_longest_song(self, length: int) -> None:
        """
        Updates the longest song seen up to this moment.
        :param length: Length of the current song
        """
        self.longest_song = max(length, self.longest_song)

    def _read_song(self, f: File) -> List[List[float]]:
        """
        Reads the One-hot representations of the notes played at the same time step.
        :param f: File object containing encoded notes for each time step in each row.
        :return: Parsed list of One-hot encoded variables for each time step.
        """
        lines = f.readlines()
        self._update_longest_song(len(lines))
        return [MusicDataset._parse_features(l) for l in lines]

    @staticmethod
    def _parse_features(line: str) -> List[float]:
        """
        Parses the One-hot representations of the notes played at one time step.
        :param line: Space separated one-hot representations of the notes.
        :return: Parsed list of One-hot encoded variables at one time step.
        """
        return list(map(float, line.strip().split()))

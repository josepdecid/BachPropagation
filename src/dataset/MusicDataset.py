import glob
import logging
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader

from utils.constants import DATASET_PATH, MAX_POLYPHONY
from utils.tensors import use_cuda
from utils.typings import File, FloatTensor


class MusicDataset(Dataset):
    def __init__(self):
        logging.info('Loading music data...')

        self.songs = []
        self.longest_song = 0
        for path in glob.glob(f'{DATASET_PATH}/*.txt'):
            with open(path, mode='r') as f:
                self.songs.append(self._read_song(f))

        self.padded_songs = self._apply_padding()

    def __getitem__(self, index):
        x = self.padded_songs[index]
        return FloatTensor(x)

    def __len__(self):
        return len(self.songs)

    def get_dataloader(self, batch_size, shuffle=False):
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def _apply_padding(self) -> List[FloatTensor]:
        padded_songs = []
        for song in self.songs:
            padded_song = torch.zeros((self.longest_song, MAX_POLYPHONY), dtype=torch.float)
            padded_song[:len(song), :] = FloatTensor(song)
            padded_songs.append(padded_song)
        return padded_songs

    def _update_longest_song(self, length: int):
        self.longest_song = max(length, self.longest_song)

    def _read_song(self, f: File) -> List[List[float]]:
        lines = f.readlines()
        self._update_longest_song(len(lines))
        return [MusicDataset._parse_features(l) for l in lines]

    @staticmethod
    def _parse_features(line):
        data = list(map(float, line.strip().split('\t')))
        frequencies = [0.0] * MAX_POLYPHONY
        frequencies[:len(data)] = data

        return [float(f) for f in frequencies]

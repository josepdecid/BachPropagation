import glob
from typing import List

from torch.utils.data import Dataset, DataLoader

from utils.tensors import use_cuda
from utils.typings import File, FloatTensor
from utils.constants import DATASET_PATH


class MusicDataset(Dataset):
    def __init__(self):
        self.songs = []
        self.longest_song = 0
        for path in glob.glob(f'{DATASET_PATH}/*.txt'):
            with open(path, mode='r') as f:
                self.songs.append(self._read_song(f))

    def __getitem__(self, index):
        x = self.songs[index]
        return FloatTensor(x)

    def __len__(self):
        return len(self.songs)

    def get_dataloader(self, batch_size, shuffle=False):
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def _apply_padding(self):
        for i in range(len(self.songs)):
            if len(self.songs[i]) < self.longest_song:
                self.songs[i] = self.songs[i] + [0.0] * (self.longest_song - len(self.songs[i]))

    def _update_longest_song(self, length: int):
        self.longest_song = max(length, self.longest_song)

    def _read_song(self, f: File) -> List[float]:
        lines = f.readlines()
        self._update_longest_song(len(lines))
        return [float(l) for l in lines]

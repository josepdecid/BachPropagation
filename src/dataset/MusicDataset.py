import glob
from typing import List

from torch.utils.data import Dataset, DataLoader

from utils.constants import PPath
from utils.tensors import use_cuda
from utils.typings import File, IntTensor


class MusicDataset(Dataset):
    def __init__(self):
        self.songs = []
        for path in glob.glob(PPath(f'/res/datasets/atis/processed/*.txt')):
            with open(path, mode='r') as f:
                self.songs.append(MusicDataset._read_song(f))

    def __getitem__(self, index):
        x = self.songs[index]
        return IntTensor(x)

    def __len__(self):
        return len(self.songs)

    def get_dataloader(self, batch_size, shuffle=False):
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def _apply_padding(self):
        pass

    @staticmethod
    def _read_song(f: File) -> List[int]:
        return [int(l) for l in f.readlines()]
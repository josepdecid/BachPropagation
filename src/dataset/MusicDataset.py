import glob
import logging

import music21 as m21
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from constants import RAW_DATASET_PATH, BATCH_SIZE, SEQUENCE_LEN
from utils.tensors import use_cuda
from utils.typings import FloatTensor


class MusicDataset(Dataset):
    """Music dataset representation, which reads, parses, adds padding and generates DataLoader for training"""

    def __init__(self):
        logging.info('Loading music data...')

        self.notes = []
        for path in glob.glob(f'{RAW_DATASET_PATH}/*.mid'):
            midi_data = m21.converter.parse(path)

            try:
                s2 = m21.instrument.partitionByInstrument(midi_data)
                notes_to_parse = s2.parts[0].recurse()
            except:
                notes_to_parse = midi_data.flat.notes

            for element in notes_to_parse:
                if isinstance(element, m21.note.Note):
                    self.notes.append(str(element.pitch))
                elif isinstance(element, m21.chord.Chord):
                    self.notes.append('.'.join(str(n) for n in element.normalOrder))

        pitch_names = sorted(set(item for item in self.notes))
        note2idx = {note: idx for idx, note in enumerate(pitch_names)}

        self.network_input = []
        self.network_output = []

        for i in range(0, len(self.notes) - SEQUENCE_LEN, 1):
            sequence_in = self.notes[i:i + SEQUENCE_LEN]
            sequence_out = self.notes[i + SEQUENCE_LEN]
            self.network_input.append([note2idx[char] for char in sequence_in])
            self.network_output.append(note2idx[sequence_out])

        self.vocab_size = len(set(self.notes))
        self.network_input = torch.tensor(self.network_input, dtype=torch.float)
        self.network_input = np.reshape(self.network_input, newshape=(-1, SEQUENCE_LEN, 1))

        # self.padded_songs = self._apply_padding()
        # self.seq_X, self.seq_Y = self._create_sequences()

    def __getitem__(self, index: int) -> FloatTensor:
        # return self.padded_songs[index]
        return self.network_input[index], self.network_output[index]

    def __len__(self) -> int:
        return len(self.network_input)

    def get_dataloader(self, shuffle=False) -> DataLoader:
        """
        Builds a data loader object with the padded songs in the dataset.
        :param shuffle: Randomize the order of the songs each time that it's called (each epoch).
        :return: PyTorch DataLoader object.
        """
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        return DataLoader(self, batch_size=BATCH_SIZE, shuffle=shuffle, **kwargs)

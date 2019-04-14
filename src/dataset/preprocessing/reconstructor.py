import logging
import os

import numpy as np
from py_midicsv import csv_to_midi, FileWriter

from constants import RESULTS_PATH, SAMPLE_TIMES, DATASET_PATH, MAX_POLYPHONY
from utils.music import freq_to_note
from utils.typings import NDArray


def store_csv_to_midi(title: str, data: str) -> str:
    """
    Parses and stores CSV data to a MIDI file.
    :param title: Title of the song (file).
    :param data: CSV string data of the song.
    """
    logging.info(f'Writing MIDI file at {RESULTS_PATH}{title}')

    file_path = f'{RESULTS_PATH}/{title}.txt'
    with open(file_path, mode='w') as f:
        f.write(data)

    with open(file_path, mode='r') as f:
        midi_data = csv_to_midi(f)
        with open(f'{RESULTS_PATH}/{title}.mid', mode='wb') as g:
            writer = FileWriter(g)
            writer.write(midi_data)

    os.remove(file_path)

    return f'{RESULTS_PATH}/{title}.mid'


def parse_data(notes_data: NDArray) -> str:
    """
    Parses song data to CSV format with MIDI event format.
    :param notes_data: List of the note played in each time step.
    :return: String containing the CSV data of the notes
    """
    logging.info('Parsing note data...')

    start_times = [0] * MAX_POLYPHONY
    current_notes = [0] * MAX_POLYPHONY
    csv_data_tracks = [[f'{idx + 2}, 0, Start_track'] for idx in range(MAX_POLYPHONY)]

    # TODO: Generalize for polyphony
    start_times = [0] * MAX_POLYPHONY
    for time_step, freqs in enumerate(notes_data):
        notes = list(map(freq_to_note, freqs))
        for idx, note in enumerate(notes):
            if note != current_notes[idx]:
                if current_notes[idx] > 0:
                    channel = idx if idx < 10 else idx + 1
                    csv_data_tracks[idx].append(
                        f'{idx + 2}, {start_times[idx] * SAMPLE_TIMES}, Note_on_c, {channel}, {note}, 64\n' +
                        f'{idx + 2}, {time_step * SAMPLE_TIMES}, Note_off_c, {channel}, {note}, 0'
                    )

                if note != 0:
                    start_times[idx] = time_step
                    current_notes[idx] = note
                else:  # Silence
                    current_notes[idx] = 0

    data_tracks = []
    for idx in range(len(csv_data_tracks)):
        csv_data_tracks[idx].append(f'{idx + 2}, 10000, End_track')
        data_tracks.append('\n'.join(csv_data_tracks[idx]))

    return '\n'.join(data_tracks)


def series_to_csv(title: str, data: NDArray) -> str:
    """
    Parses the output of the GAN generator to a CSV with the required format.
    :param title: Title of the song.
    :param data: Data output by the generator.
    :return: String with the CSV data.
    """
    logging.info('Converting to CSV...')

    header = [f'0, 0, Header, 1, {MAX_POLYPHONY}, 384',
              '1, 0, Start_track',
              f'1, 0, Title_t, "{title}"',
              '1, 0, Time_signature, 4, 2, 24, 8',
              '1, 0, Tempo, 550000',
              '1, 0, End_track',
              '2, 0, Start_track',
              '2, 0, Text_t, "RH"']
    header = '\n'.join(header)

    footer = ['0, 0, End_of_file']
    footer = '\n'.join(footer)

    return f'{header}\n{parse_data(data)}\n{footer}'


def reconstruct_midi(title: str, data: NDArray) -> str:
    """
    Parses the output of the GAN generator to a MIDI file.
    :param title: Title of the generated song.
    :param data: Data output by the generator.
    """
    logging.info(f'Creating {title}')

    # TODO: Generalize for polyphony
    csv_data = series_to_csv(title=title, data=data)
    return store_csv_to_midi(title=title, data=csv_data)

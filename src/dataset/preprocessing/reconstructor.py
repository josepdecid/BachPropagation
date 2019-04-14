import logging
import os

import numpy as np
from py_midicsv import csv_to_midi, FileWriter

from constants import RESULTS_PATH, DATASET_PATH, MAX_POLYPHONY, MAX_FREQ_NOTE
from utils.music import freq_to_note
from utils.typings import NDArray


def store_csv_to_midi(title: str, csv_data: str) -> str:
    """
    Parses and stores CSV data to a MIDI file.
    :param title: Title of the song (file).
    :param csv_data: CSV string data of the song.
    """
    logging.info(f'Writing MIDI file at {RESULTS_PATH}{title}')

    file_path = f'{RESULTS_PATH}/{title}.csv'
    with open(file_path, mode='w') as f:
        f.write(csv_data)

    with open(file_path, mode='r') as f:
        midi_data = csv_to_midi(f)
        with open(f'{RESULTS_PATH}/{title}.mid', mode='wb') as g:
            writer = FileWriter(g)
            writer.write(midi_data)

    # os.remove(file_path)

    return f'{RESULTS_PATH}/{title}.mid'


def parse_data(notes_data: NDArray) -> str:
    """
    Parses song data to CSV format with MIDI event format.
    :param notes_data: List of the note played in each time step.
    :return: String containing the CSV data of the notes
    """
    logging.info('Parsing note data...')

    last_time, end_time = 0, 0
    last_time_track_played = [0] * MAX_POLYPHONY
    csv_data_tracks = [[f'{idx + 2}, 0, Start_track'] for idx in range(MAX_POLYPHONY)]

    for note_data in notes_data:
        note = freq_to_note(float(note_data[0]) * MAX_FREQ_NOTE)
        duration = int(note_data[1])
        time_since_previous = int(note_data[2])

        start_time = last_time + time_since_previous
        end_time = start_time + duration
        last_time = start_time

        track_idx = None
        for idx in range(len(last_time_track_played)):
            if end_time >= last_time_track_played[idx]:
                track_idx = idx
                last_time_track_played[idx] = end_time
                break

        if track_idx is None:
            continue

        channel = track_idx if track_idx < 10 else track_idx + 1

        csv_data_tracks[track_idx].append(
            f'{track_idx + 2}, {start_time}, Note_on_c, {channel}, {note}, 64\n' +
            f'{track_idx + 2}, {end_time}, Note_off_c, {channel}, {note}, 0'
        )

    data_tracks = []
    for idx in range(len(csv_data_tracks)):
        if len(csv_data_tracks[idx]) == 1:
            continue
        csv_data_tracks[idx].append(f'{idx + 2}, {last_time_track_played[idx]}, End_track')
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

    header = [f'0, 0, Header, 1, {MAX_POLYPHONY + 2}, 384',
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


def reconstruct_midi(title: str, raw_data: NDArray) -> str:
    """
    Parses the output of the GAN generator to a MIDI file.
    :param title: Title of the generated song.
    :param raw_data: Data output by the generator.
    """
    logging.info(f'Creating {title}')

    csv_data = series_to_csv(title=title, data=raw_data)
    return store_csv_to_midi(title=title, csv_data=csv_data)


with open(f'{DATASET_PATH}/BWV_810_01_midi.txt') as f:
    data = f.read().strip().split('\n')
    data = np.array(list(map(lambda x: x.split(), data)))
    reconstruct_midi('TESTTTTT', data)

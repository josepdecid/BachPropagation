import logging
import os
from typing import List

from py_midicsv import csv_to_midi, FileWriter

from constants import RESULTS_PATH, SAMPLE_TIMES, MIN_NOTE


def store_csv_to_midi(title: str, data: str) -> str:
    """
    Parses and stores CSV data to a MIDI file.
    :param title: Title of the song (file).
    :param data: CSV string data of the song.
    """
    logging.info(f'Writing MIDI file at {RESULTS_PATH}')

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


def parse_data(data: List[int]) -> str:
    """
    Parses song data to CSV format with MIDI event format.
    :param data: List of the note played in each time step.
    :return: String containing the CSV data of the notes
    """
    logging.info('Parsing note data...')

    current_note = None
    csv_data = []

    # TODO: Generalize for polyphony
    for time_step, note in enumerate(data):
        if note != current_note:
            if current_note is not None:
                csv_data.append(f'2, {start_time * SAMPLE_TIMES}, Note_on_c, 0, {note + MIN_NOTE}, 64\n'
                                f'2, {time_step * SAMPLE_TIMES}, Note_off_c, 0, {note + MIN_NOTE}, 0')

            if note != 0:
                start_time = time_step
                current_note = note
            else:  # Silence
                current_note = None

    return '\n'.join(csv_data)


def series_to_csv(title: str, data: List[int]) -> str:
    """
    Parses the output of the GAN generator to a CSV with the required format.
    :param title: Title of the song.
    :param data: Data output by the generator.
    :return: String with the CSV data.
    """
    logging.info('Converting to CSV...')

    header = ['0, 0, Header, 1, 2, 480',
              '1, 0, Start_track',
              '1, 0, Title_t, "' + title + '"',
              '1, 0, Time_signature, 4, 2, 24, 8',
              '1, 0, Tempo, 500000',
              '1, 0, End_track',
              '2, 0, Start_track',
              '2, 0, Text_t, "RH"']
    header = '\n'.join(header)

    footer = ['2, 4800, End_track', '0, 0, End_of_file']
    footer = '\n'.join(footer)

    return f'{header}\n{parse_data(data)}\n{footer}'


def reconstruct_midi(title: str, data: List[int]) -> str:
    """
    Parses the output of the GAN generator to a MIDI file.
    :param title: Title of the generated song.
    :param data: Data output by the generator.
    """
    logging.info(f'Creating {title}')

    # TODO: Generalize for polyphony
    csv_data = series_to_csv(title=title, data=data)
    return store_csv_to_midi(title=title, data=csv_data)

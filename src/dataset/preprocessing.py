import glob
import logging
from typing import List

from py_midicsv import midi_to_csv

from dataset.Music import Song
from utils.constants import RAW_DATASET_PATH


def csv_cleaner(data: List[str]) -> List[Song]:
    """
    Reduces useless information from MIDI tracks, extracting each track of the song and obtaining each note time
    boundaries, along with the note encoding and its normalized velocity.
    :param data: List of CSV data for all files in path.
    :return: List of tuples (note_start, note_end, note_code, velocity) for each played note and for each track
    """
    raise NotImplementedError


def csv_to_series(song: Song, time_step=10) -> List[List[int]]:
    ts = 0
    track_time_indices = [0] * song.number_tracks

    # Get max time of last Note data of all tracks.
    max_time = song.max_time
    series_data = [[]] * max_time // time_step

    while ts < max_time:
        for track_idx, track_time_idx in enumerate(track_time_indices):
            note_data = song.get_track(track_idx).get_note_data(track_time_idx)
            if note_data.is_playing(ts):
                series_data[(ts // time_step) - 1].append(note_data.note)
        ts += time_step

    return series_data


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    logging.info('Converting MIDI to CSV...')
    files = glob.glob(f'{RAW_DATASET_PATH}/*.mid')
    csv_data = list(map(midi_to_csv, files))
    logging.info(f'Converting {len(files)} files')

    logging.info('Cleaning CSV files...')
    csv_preprocessed = list(map(csv_cleaner, csv_data))

    logging.info('Converting information to time series...')
    time_series = list(map(csv_to_series, csv_preprocessed))

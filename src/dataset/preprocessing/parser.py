import glob
import logging
import re
from typing import List, Dict

from py_midicsv import midi_to_csv
from tqdm import tqdm

from constants import RAW_DATASET_PATH, DATASET_PATH, SAMPLE_TIMES, MAX_POLYPHONY
from dataset.Music import Song, Track, NoteData
from utils.music import note_to_freq


def csv_cleaner(data: List[str]) -> Song:
    """
    Reduces useless information from MIDI tracks, extracting each track of the song and obtaining each note time
    boundaries, along with the note encoding and its normalized velocity.
    :param data: List of CSV data for all files in path.
    :return: List of tuples (note_start, note_end, note_code, velocity) for each played note and for each track
    """
    # Discard header rows
    idx = 0
    for idx, row in enumerate(data):
        if 'Note_on_c' in row:
            break

    tracks = []
    notes_data = []
    current_notes: Dict[int, NoteData] = {}

    for idx, row in enumerate(data[idx:]):
        row = row.strip()

        # Note information events
        if re.match(r'^\d+, \d+, Note_(on|off)_c, \d+, \d+, \d+$', row):
            track, note_time, event, channel, note, velocity = row.split(', ')

            # Note ends event (event if off or 0 velocity)
            # We add missing end attribute and push to list of data
            if event == 'Note_off_c' or (event == 'Note_on_c' and velocity == '0'):
                # TODO: Review strange missing starts
                if note in current_notes:
                    current_notes[note].note_end = int(note_time)
                    notes_data.append(current_notes[note])
                    del current_notes[note]

            # Note starts event. Gets start_time, note and velocity information.
            elif event == 'Note_on_c':
                current_notes[note] = NoteData(int(note_time), 0, int(note), int(velocity))

        # Push and start new track if end of track reached
        elif re.match(r'^\d+, \d+, End_track$', row) and len(notes_data):
            tracks.append(Track(notes_data))
            notes_data = []
            current_notes = {}

    return Song(tracks)


def csv_to_series(song: Song) -> List[List[int]]:
    """

    :param song:
    :return:
    """
    # Index of current treated element for each track.
    track_time_indices = [0] * song.number_tracks

    # Get max time of last Note data of all tracks.
    max_time = song.max_time // SAMPLE_TIMES
    series_data = [[] for _ in range(max_time)]

    time_idx = 0
    while time_idx < max_time:
        for track_idx, track_time_idx in enumerate(track_time_indices):
            # Skip if track already finished
            if track_time_indices[track_idx] >= song.get_track(track_idx).len_track:
                continue

            # Add note to current time step if it's being played
            note_data = song.get_track(track_idx).get_note_data(track_time_idx)
            if note_data.is_playing(time_idx * SAMPLE_TIMES):
                # Add as maximum MAX_POLYPHONY notes for each step
                if len(series_data[time_idx]) >= MAX_POLYPHONY:
                    continue
                series_data[time_idx].append(note_data.note)

            # Check if note won't be played on next time step
            if not note_data.is_playing((time_idx + 1) * SAMPLE_TIMES):
                track_time_indices[track_idx] += 1

        time_idx += 1

    return series_data


def notes_to_freq(series: List[List[int]]) -> List[List[float]]:
    notes_freqs = []
    for notes in series:
        freqs = []
        for note in notes:
            freqs.append(note_to_freq(note))
        notes_freqs.append(freqs)
    return notes_freqs


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    logging.info('Converting MIDI to CSV...')
    files = glob.glob(f'{RAW_DATASET_PATH}/*.mid')
    csv_data = list(map(midi_to_csv, tqdm(files, ncols=150)))
    logging.info(f'Converting {len(files)} files')

    logging.info('Cleaning CSV files...')
    csv_preprocessed = list(map(csv_cleaner, tqdm(csv_data, ncols=150)))

    logging.info('Converting information to time series...')
    time_series = list(map(csv_to_series, tqdm(csv_preprocessed, ncols=150)))

    logging.info('One-hot encoding notes...')
    input_notes = list(map(notes_to_freq, tqdm(time_series, ncols=150)))

    logging.info('Writing note features ...')
    for path, time_steps in tqdm(zip(files, input_notes), ncols=150):
        file = path.split('/')[-1][:-4] + '.txt'
        with open(f'{DATASET_PATH}/{file}', mode='w') as f:
            for ts in time_steps:
                f.write(' '.join(map(str, ts)) + '\n')

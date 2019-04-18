import glob
import logging
import re
from typing import List, Dict, Tuple

from py_midicsv import midi_to_csv
from tqdm import tqdm

from constants import RAW_DATASET_PATH, DATASET_PATH, MAX_POLYPHONY, NORMALIZE_FREQ, NORMALIZE_VEL
from dataset.Music import Song, Track, NoteData


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
            # Skip percussion channel
            if channel == 9:
                continue

            # Note ends event (event if off or 0 velocity)
            # We add missing end attribute and push to list of data
            if event == 'Note_off_c' or (event == 'Note_on_c' and velocity == '0'):
                # MIDI specification allows to set off a note multiple times
                if note in current_notes:
                    current_notes[note].note_end = int(note_time)
                    notes_data.append(current_notes[note])
                    del current_notes[note]

            # Note starts event. Gets start_time, note and velocity information.
            elif event == 'Note_on_c' and note not in current_notes:
                current_notes[note] = NoteData(int(note_time), 0, int(note), int(velocity))

        # Push and start new track if end of track reached
        elif re.match(r'^\d+, \d+, End_track$', row) and len(notes_data):
            tracks.append(Track(notes_data))
            notes_data = []
            current_notes = {}
            if len(tracks) >= MAX_POLYPHONY:
                break

    return Song(tracks)


def csv_to_series(song: Song) -> List[Tuple[float, float, int, int]]:
    """
    Converts song data to time series data.
    :param song: Song data.
    :return: Sequence of (Frequency, Velocity, Duration, TimeSincePrevious).
    """
    ts_data = []
    last_start = None
    sorted_notes = song.get_notes_in_start_order()

    for note in sorted_notes:
        time_since_last = note.note_start - last_start if last_start is not None else 0
        last_start = note.note_start

        ts_data.append((
            note.norm_freq if NORMALIZE_FREQ else note.freq,
            note.norm_vel if NORMALIZE_VEL else note.velocity,
            note.duration,
            time_since_last
        ))

    return ts_data


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    logging.info('Converting MIDI to CSV...')
    files = glob.glob(f'{RAW_DATASET_PATH}/*.mid')
    csv_data = list(map(midi_to_csv, tqdm(files, ncols=150)))
    logging.info(f'Converting {len(files)} files')

    for path, d in zip(files, csv_data):
        file = path.split('/')[-1][:-4] + '.csv'
        with open(f'{DATASET_PATH}/{file}', mode='w') as f:
            f.write(''.join(d))

    logging.info('Cleaning CSV files...')
    csv_preprocessed = list(map(csv_cleaner, tqdm(csv_data, ncols=150)))

    logging.info('Converting information to time series...')
    time_series = list(map(csv_to_series, tqdm(csv_preprocessed, ncols=150)))

    logging.info('Writing note features ...')
    for path, time_steps in zip(files, time_series):
        file = path.split('/')[-1][:-4] + '.txt'
        with open(f'{DATASET_PATH}/{file}', mode='w') as f:
            for ts in time_steps:
                f.write(' '.join(map(str, ts)) + '\n')

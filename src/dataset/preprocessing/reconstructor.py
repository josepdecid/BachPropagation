from py_midicsv import csv_to_midi, FileWriter

from utils.constants import RESULTS_PATH, DATASET_PATH, PPath

path = f'{DATASET_PATH}/ala.csv'
midi_data = csv_to_midi(path)

with open(PPath('/res/results/ala.mid'), mode='wb') as f:
    writer = FileWriter(f)
    writer.write(midi_data)


def parse_data(data):
    for d in


def generate_csv(title, data):
    csv_data = ['0, 0, Header, 1, 2, 480',
                '1, 0, Start_track',
                '1, 0, Title_t, "' + title + '"',
                '1, 0, Text_t, "GAN generated music"',
                '1, 0, Copyright_t, "This file is in the public domain"',
                '1, 0, Time_signature, 4, 2, 24, 8',
                '1, 0, Tempo, 500000',
                '1, 0, End_track',
                '2, 0, Start_track',
                '2, 0, Instrument_name_t, "Acoustic Grand Piano"',
                '2, 0, Program_c, 1, 19']
    csv_data = '\n'.join(csv_data)

    csv_data.append(parse_data(data))

    return csv_data

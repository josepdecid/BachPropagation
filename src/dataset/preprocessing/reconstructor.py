from py_midicsv import csv_to_midi, FileWriter

from utils.constants import RESULTS_PATH, DATASET_PATH, PPath

path = f'{DATASET_PATH}/ala.csv'
midi_data = csv_to_midi(path)

with open(PPath('/res/results/ala.mid'), mode='wb') as f:
    writer = FileWriter(f)
    writer.write(midi_data)
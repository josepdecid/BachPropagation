import glob
import logging

import py_midicsv

from utils.constants import RAW_DATASET_PATH, DATASET_PATH

logging.getLogger().setLevel(logging.INFO)

for file in glob.glob(f'{RAW_DATASET_PATH}/*.mid'):
    file_name = file.split('/')[-1][:-4]  # Same file name with csv extension
    logging.info(f'Parsing {file_name}')

    data = py_midicsv.midi_to_csv(file)
    with open(f'{DATASET_PATH}/{file_name}.csv', mode='w') as f:
        f.write(''.join(data))

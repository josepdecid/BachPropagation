import os
from pathlib import Path

SOURCE_MIDI_URLS = ['http://www.jsbach.es/bbdd/index01_20.htm']

PROJECT_PATH = os.getenv('BACHPROPAGATION_ROOT_PATH')
DATASET_PATH = os.getenv('BACHPROPAGATION_ROOT_PATH') + '/res/dataset/processed'
RAW_DATASET_PATH = os.getenv('BACHPROPAGATION_ROOT_PATH') + '/res/dataset/raw'

PPath = lambda p: Path(PROJECT_PATH + p)
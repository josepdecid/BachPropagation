import os
from pathlib import Path

SOURCE_MIDI_URLS = ['http://www.jsbach.es/bbdd/index01_20.htm']

PROJECT_PATH = os.getenv('BACHPROPAGATION_ROOT_PATH')
DATASET_PATH = os.getenv('BACHPROPAGATION_ROOT_PATH') + '/res/dataset/processed'
RAW_DATASET_PATH = os.getenv('BACHPROPAGATION_ROOT_PATH') + '/res/dataset/raw'
RESULTS_PATH = os.getenv('BACHPROPAGATION_ROOT_PATH') + '/res/results/'
CHECKPOINTS_PATH = os.getenv('BACHPROPAGATION_ROOT_PATH') + '/res/checkpoints'

PPath = lambda p: Path(PROJECT_PATH + p)

CKPT_STEPS = 100

NUM_NOTES = 88
MIN_NOTE = 21

# HYPERPARAMETERS

EPOCHS = 100
BATCH_SIZE = 8
MAX_POLYPHONY = 1
SAMPLE_TIMES = 10

LR_G = 0.1
L2_G = 0.25
HIDDEN_DIM_G = 50
BIDIRECTIONAL_G = False
LAYERS_G = 1

LR_D = 0.1
L2_D = 0.25
HIDDEN_DIM_D = 50
BIDIRECTIONAL_D = True
LAYERS_D = 1

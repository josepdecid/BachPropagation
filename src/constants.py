import os
import numpy as np
from pathlib import Path

SOURCE_MIDI_URLS = ['http://www.jsbach.es/bbdd/index01_20.htm']

PROJECT_PATH = os.getenv('BACHPROPAGATION_ROOT_PATH')
DATASET_PATH = os.getenv('BACHPROPAGATION_ROOT_PATH') + '/res/dataset/processed'
RAW_DATASET_PATH = os.getenv('BACHPROPAGATION_ROOT_PATH') + '/res/dataset/raw'
RESULTS_PATH = os.getenv('BACHPROPAGATION_ROOT_PATH') + '/res/results/'
CHECKPOINTS_PATH = os.getenv('BACHPROPAGATION_ROOT_PATH') + '/res/checkpoints'
LOG_PATH = os.getenv('BACHPROPAGATION_ROOT_PATH') + '/res/log'

PPath = lambda p: Path(PROJECT_PATH + p)

PLOT_COL = {
    'G': np.array([[75, 159, 56]]),  # Green
    'D': np.array([[220, 65, 51]])  # Red
}

SAMPLE_STEPS = 1
CKPT_STEPS = 100
INPUT_FEATURES = 3

MIN_FREQ_NOTE = 27.5
MAX_FREQ_NOTE = 4186.0
NUM_NOTES = 88
MIN_NOTE = 21

FLAGS = {
    'viz': True
}

###################
# HYPERPARAMETERS #
###################

EPOCHS = 10000
BATCH_SIZE = 16
MAX_POLYPHONY = 12
NORMALIZE_FREQ = False
T_LOSS_BALANCER = 0.7

# Generator
LR_G = 0.4
LR_PAT_G = 50
L2_G = 0.25
HIDDEN_DIM_G = 50
BIDIRECTIONAL_G = False
PRETRAIN_G = 0
TYPE_G = 'LSTM'
LAYERS_G = 1

# Discriminator
LR_D = 0.4
LR_PAT_D = 50
L2_D = 0.25
HIDDEN_DIM_D = 50
BIDIRECTIONAL_D = True
PRETRAIN_D = 0
TYPE_D = 'LSTM'
LAYERS_D = 1

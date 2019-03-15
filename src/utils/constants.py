import os

SOURCE_MIDI_URLS = ['http://www.jsbach.es/bbdd/index01_20.htm']
XPATH_URL_FILTER = '//a[re:test(@target, "_repro") and re:test(@href, ".mid$")]//@href'

PROJECT_PATH = os.getenv('BACHPROPAGATION_ROOT_PATH')
RAW_DATASET_PATH = f'{PROJECT_PATH}/res/dataset/raw'
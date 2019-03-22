import logging
import shutil
from pathlib import Path
from urllib import request

from bs4 import BeautifulSoup

from utils.constants import RAW_DATASET_PATH, SOURCE_MIDI_URLS

logging.getLogger().setLevel(logging.INFO)

for url in SOURCE_MIDI_URLS:
    page = request.urlopen(url)
    soup = BeautifulSoup(page, 'html.parser')
    download_link_tags = soup.find_all('a', attrs={'target': '_repro'}, href=True)
    download_links = list(filter(lambda x: x[-4:] == '.mid', map(lambda x: x['href'], download_link_tags)))

    base_url = '/'.join(SOURCE_MIDI_URLS[0].split('/')[:-1])
    for link in download_links:
        file_name = link.split('/')[-1]
        download_url = f'{base_url}/{link}'

        with request.urlopen(download_url) as response:
            with open(Path(f'{RAW_DATASET_PATH}/{file_name}'), mode='wb') as f:
                shutil.copyfileobj(response, f)

        logging.info(f'Downloaded and saved {file_name}')

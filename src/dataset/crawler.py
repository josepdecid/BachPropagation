import shutil
from urllib import request

import scrapy
from scrapy import Selector
from scrapy.crawler import CrawlerProcess

from utils.constants import RAW_DATASET_PATH, SOURCE_MIDI_URLS, XPATH_URL_FILTER


class MIDICrawler(scrapy.Spider):
    name = 'MIDICrawler'

    def start_requests(self):
        for url in SOURCE_MIDI_URLS:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        download_paths = Selector(response=response, type='html').xpath(XPATH_URL_FILTER).extract()

        base_url = '/'.join(SOURCE_MIDI_URLS[0].split('/')[:-1])
        for path in download_paths:
            file_name = path.split('/')[-1]
            download_url = f'{base_url}/{path}'

            with request.urlopen(download_url) as response:
                with open(f'{RAW_DATASET_PATH}/{file_name}', mode='wb') as f:
                    shutil.copyfileobj(response, f)

            self.logger.info(f'Downloaded and saved {file_name}')


process = CrawlerProcess()
process.crawl(MIDICrawler)
process.start(stop_after_crawl=False)

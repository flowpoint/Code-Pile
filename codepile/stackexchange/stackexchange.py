import os
import itertools
from datetime import datetime
import json
import shutil
import gc
from functools import partial
from more_itertools import chunked
import multiprocessing as mp

from lxml import etree
from tqdm import tqdm
from lm_dataformat import Archive
import internetarchive as ia
import py7zr

from codepile.dataset import DatasetInfo, DatasetSources, RawDataset, Scraper, Processor, Analyser, Dataset

# todo
STACKEXCHANGEINFO = DatasetInfo(
        id='StackExchange',
        description='',
        data_end=datetime(2022,1,1),
        data_start=10,
        size=10,
        storage_format='tar',
        #storage_uri='/root',
        cpu_hours=1,
        gpu_hours=1,
        ram_requirements=1,
        tempfile_requirement=1,
        source_uri='https://archive.org/details/stackexchange',
        dataset_pros='l',
        dataset_cons='l',
        languages=[''],
        coding_languages=[''],
        modalities=['discussion'],
        source_license='gpl',
        source_citation='this',
        data_owner='me',
        contributers=['me']
        )


class StackExchangeScraper(Scraper):
    def scrape(self) -> RawDataset:
        item = ia.get_item('stackexchange')
        metadata = item.metadata
        ia.download('stackexchange', checksum=True, verbose=True, destdir=self.config.raw_data_dir)

        def to_uri(x):
            return 'file://'+os.path.join(self.config.raw_data_dir, 'stackexchange', x)

        storage_uris = list(map(to_uri,                
            os.listdir(os.path.join(self.config.raw_data_dir, 'stackexchange'))
            ))
        ds = RawDataset(storage_uris=storage_uris, metadata=str(metadata))
        return ds

class StackExchangeCodeProcessor(Processor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # files specific to stackoverflow
        self.batch_size=100000
        self.stackoverflow_files = [
                'stackoverflow.com-Badges.7z',
                'stackoverflow.com-Comments.7z',
                'stackoverflow.com-PostHistory.7z',
                'stackoverflow.com-PostLinks.7z',
                'stackoverflow.com-Posts.7z',
                'stackoverflow.com-Tags.7z',
                'stackoverflow.com-Users.7z',
                'stackoverflow.com-Votes.7z'
                ]

        # the all files present in the dump for each subdomain
        # select the tables that we're interested in
        self.target_files = [
                #'Badges.xml',
                #'Comments.xml',
                #'PostHistory.xml',
                #'PostLinks.xml',
                'Posts.xml',
                #'Tags.xml',
                'Users.xml',
                #'Votes.xml',
                ]

        self.target_schema_python = {
                'Users.xml': user_schema_python,
                'Posts.xml': post_schema_python
                }

        self.target_schema = {
                'Users.xml': user_schema,
                'Posts.xml': post_schema
                #'Comments.xml',
                }

    def save_to_jsonl(self, dict_):
        raise NotImplementedError

    def format(self, dict_) -> dict:
        raise NotImplementedError

    def process(self, raw_data: RawDataset, *args, **kwargs):
        raise NotImplementedError


    def extract_dumpfile(self, domain: str, input_uri: str):
        raise NotImplementedError

    def save_intermediate(self, domain: str, target: str):
        raise NotImplementedError

class StackExchangeDataset(Dataset):
    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.scraper = StackExchangeScraper(self.config, self.id)
        self.processor = StackExchangeCodeProcessor(self.config, self.id)

    @property
    def info(self):
        return STACKEXCHANGEINFO

    @property
    def id(self):
        return "StackExchange"

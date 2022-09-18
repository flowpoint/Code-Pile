import os
import itertools
from datetime import datetime

import multiprocessing as mp

import internetarchive as ia
from pydantic import BaseModel
import py7zr

import urllib
from functools import partial

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


class Comment(BaseModel):
    user_id: str
    score: int
    creation_date: datetime
    text: str


class User(BaseModel):
    user_id: str
    reputation: int
    upvotes: int
    downvotes: int
    profileviews: int
    creationdate: datetime
    # this is not pure text
    aboutme: str


class Post(BaseModel):
    # ids
    post_id: int
    user_id: str

    # main post text
    title: str
    text: str
    comments: list[Comment]
    
    score: int
    tags: list[str]
    favourite_count: int
    view_count: int
    creation_date: datetime
    last_edit_date: datetime


# this should represent a single document/jsonl line
class StackExchangeDoc(BaseModel):
    question: Post
    answers: list[Post]
    accepted_answer_post_id: int
    users: list[User]


def extract(output_dir, input_file):
    input_file = input_file.replace('file://', '')
    path = os.path.join(output_dir, os.path.basename(input_file))

    with py7zr.SevenZipFile(input_file, mode='r') as z:
        z.extractall(path=path)


class StackExchangeCodeProcessor(Processor):
    def process(self, raw_data: RawDataset, *args, **kwargs):
        data_files = raw_data.storage_uris
        data_files = list(filter(lambda x: x.endswith('.7z'), data_files))

        # stackoverflow sites are spread over multiple 7z. files
        stackoverflow_files = [
                'stackoverflow.com-Badges.7z',
                'stackoverflow.com-Comments.7z',
                'stackoverflow.com-PostHistory.7z',
                'stackoverflow.com-PostLinks.7z',
                'stackoverflow.com-Posts.7z',
                'stackoverflow.com-Tags.7z',
                'stackoverflow.com-Users.7z',
                'stackoverflow.com-Votes.7z'
                ]

        stackoverflow_data_uris = list(filter(
                lambda x: os.path.basename(x) in stackoverflow_files,
                data_files
                ))

        # remove the singular stackoverflow files
        for u in stackoverflow_data_uris:
            data_files.remove(u)

        data_files = list(map(lambda x: [x], data_files))

        # readd the stackoverflow files as a group
        data_files.insert(0, stackoverflow_data_uris)

        # parallel processing the subparts
        # likely wanna sort, to process the biggest files first
        # because processing can be sequential and linear with filesize
        with mp.Pool(12) as p:
            p.map(self.process_subdomain, uris[0:1], chunksize=1)

    def process_subdomain(self, uris: list[str]):
        assert isinstance(uris, list)

        # handle the split stackoverflow files separately
        if len(uris) > 1:
            d = os.path.join(self.config.tmpdir, self.dataset_id, 'stackoverflow.com.7z')
            os.makedirs(d, exist_ok=True)

            def extract(inpath, outpath):
                with py7zr.SevenZipFile(inpath, mode='r') as z:
                    z.extractall(path=outpath)

            uri = uri.replace('file://', '')
            input_files = [uri.replace('file://', '') for uri in uris]

            output_dir = os.path.join(dirpath, os.path.basename(uri))

            # extract is a separate funcion because of pickeling
            # parallel because zip is single core and slow
            with mp.Pool(12) as p:
                p.map(partial(extract, output_dir), input_files, chunksize=1)

        else:
            uri = uris[0]
            uri = uri.replace('file://', '')
            d = os.path.join(self.config.tmpdir, self.dataset_id, os.path.basename(uri))
            os.makedirs(d, exist_ok=True)

            with py7zr.SevenZipFile(uri, mode='r') as z:
                z.extractall(path=os.path.join(d, os.path.basename(uri)))


        files = [
                'Badges.xml',
                'Comments.xml',
                'PostHistory.xml',
                'PostLinks.xml',
                'Posts.xml',
                'Tags.xml',
                'Users.xml',
                'Votes.xml',
                ]

        # verify
        for subdomain in os.listdir(os.path.join(self.config.tmpdir, self.dataset_id)):
            if not subdomain.endswith(".7z"):
                continue
            else:
                for f in files:
                    p = os.path.join(self.config.tmpdir, self.dataset_id, subdomain))
                    assert f in os.listdir(p), \
                            f'file: {f} missing in folder {p} \
                            during processing stackexchange subdomain'



class StackExchangeDataset(Dataset):
    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.scraper = StackExchangeScraper(self.config, self.id)
        self.processor = StackExchangeCodeProcessor(self.config, self.id)

    '''
    def download(self, *args, **kwargs):
        return super().download()

    def process(self, *args, **kwargs):
        return super().process()
    '''

    @property
    def info(self):
        return STACKEXCHANGEINFO

    @property
    def id(self):
        return "StackExchange"

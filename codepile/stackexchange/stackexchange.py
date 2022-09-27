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
        storage_format='.jsonl.zst',
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


def parse_xml(source_xml, output_dirpath) -> dict :
    # use lxml because pyarrow readxml has trouble with types
    for event, element in etree.iterparse(source_xml, events=('end',), tag='row'):
        j = dict(element.attrib)
        yield j

        # cleanup this element, and parents, to save memory
        # https://stackoverflow.com/questions/7171140/using-python-iterparse-for-large-xml-files
        element.clear(keep_tail=True)
        while element.getprevious() is not None:
            del element.getparent()[0]


class StackExchangeCodeProcessor(Processor):
    def __init__(self, *args, **kwargs):
        #super().__init__(self, config, *args, **kwargs)
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

    def save_to_jsonl(self, dict_):
        raise NotImplementedError

    def format(self, dict_) -> dict:
        raise NotImplementedError

    def join_and_export(self, domain: str, postsfile, disk_offload=True):
        raise NotImplementedError

    def process(self, raw_data: RawDataset, *args, **kwargs):
        data_files = raw_data.storage_uris
        data_files = list(filter(lambda x: x.endswith('.7z'), data_files))

        def get_domain(x):
            fname = os.path.basename(x)
            if fname in self.stackoverflow_files:
                domain = 'stackoverflow.com'
            else:
                domain = fname.replace('.7z', '')
            return domain

        def group_by_domain(data_files):
            g = dict()
            for k, v in itertools.groupby(
                    sorted(data_files, key=get_domain), 
                    key=get_domain
                    ):
                # warning, itertools shares the iterator
                # so listing the grouper v consumes the values
                # use vals or g for further computation on the group
                vals = list(v)
                g[k] = vals

            return g


        g = group_by_domain(data_files)
        parallel = True
        #parallel = False
        async_results = dict()
        print('running parallel processing')
        with mp.Pool() as p:
            # extract all the needed dumps
            for domain, dumps in g.items():
                async_results[domain] = dict()
                for dump in dumps:
                    if parallel:
                        c = p.apply_async(self.extract_dumpfile, (domain, dump))
                        async_results[domain][dump] = c
                    else:
                        self.extract_dumpfile(domain, dump)


            # wait and then transcode to arrow
            for domain in g.keys():
                # wait until all domain targets are extracted
                for dump, result in async_results[domain].items():
                    result.wait()
                    assert result.successful()


    def extract_dumpfile(self, domain: str, input_uri: str):
        print(f'extracting {domain} {input_uri}')
        dataset_tmpdir = os.path.join(self.config.tmpdir, self.dataset_id)
        input_filepath = input_uri.replace('file://', '')
        input_filename = os.path.basename(input_filepath)

        output_dirpath = os.path.join(dataset_tmpdir, domain)

        # move stackoverflow target file to output_dirpath
        if domain == 'stackoverflow.com':
            os.makedirs(output_dirpath, exist_ok=True)
            target = input_filename.replace('stackoverflow.com-', '').replace('.7z', '.xml')
            # skip if already present
            if os.path.isfile(os.path.join(output_dirpath, target)):
                return

            if target in self.target_files:
                with py7zr.SevenZipFile(input_filepath, mode='r') as z:
                    z.extractall(path=output_dirpath)

        else:
            # skip if already present
            if os.path.isdir(output_dirpath):
                return

            os.makedirs(output_dirpath, exist_ok=True)
            # all targets are in the same zip, so we unzip
            with py7zr.SevenZipFile(input_filepath, mode='r') as z:
                z.extractall(path=output_dirpath)


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

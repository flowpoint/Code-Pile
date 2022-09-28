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

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.dataset as ds

from codepile.dataset import DatasetInfo, DatasetSources, RawDataset, Scraper, Processor, Analyser, Dataset



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

        # key, python type, pyarrow type
        post_types = (
            ('Id', int, pa.uint32),
            ('OwnerUserId', int, pa.int32),
            ('ParentId', int, pa.uint32),
            ('PostTypeId', int, pa.uint8),
            ('Score', int, pa.int32), # can be negative
            ('Title', str, pa.large_string),
            ('Body', str, pa.large_string),
            ('Tags', str, pa.large_string),
            ('FavoriteCount', int, pa.uint32),
            ('CreationDate', datetime.fromisoformat, pa.date64),
            ('LastEditDate', datetime.fromisoformat, pa.date64),
            )

        user_types = (
            ('Id', int, pa.int32),
            ('Reputation', int, pa.int32),
            ('DisplayName', str, pa.large_string),
            ('Views', int, pa.uint32),
            ('UpVotes', int, pa.uint32),
            ('DownVotes', int, pa.uint32),
            ('AccountId', int, pa.int32),
            ('ProfileImageUrl', str, pa.large_string),

            ('CreationDate', datetime.fromisoformat, pa.date64),
            ('LastAccessDate', datetime.fromisoformat, pa.date64),
            )

        post_schema = pa.schema([(k,a()) for k,p,a in post_types])
        post_schema_python = {k:p for k,p,a in post_types}

        user_schema = pa.schema([(k,a()) for k,p,a in user_types])
        user_schema_python = {k:p for k,p,a in user_types}

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

            # wait 
            for domain in g.keys():
                # wait until all domain targets are extracted
                for dump, result in async_results[domain].items():
                    result.wait()
                    assert result.successful()

        gc.collect()

        # transcode to arrow
        for domain in g.keys():
             # queue transcoding to intermediate format
            for target in self.target_files:
                self.save_intermediate(domain, target)
                gc.collect()

        # join tables to dicts
        for domain in g.keys():
            # only need to offload the biggest domains
            disk_offload = domain == 'stackoverflow.com'
            # queue transcoding to intermediate format
            self.join_and_export(domain, disk_offload)
            gc.collect()
            
        print('done')


    def parse_xml(self, source_xml, output_dirpath) -> dict :
        # use lxml because pyarrow readxml has trouble with types
        for event, element in etree.iterparse(source_xml, events=('end',), tag='row'):
            j = dict(element.attrib)
            yield j

            # cleanup this element, and parents, to save memory
            # https://stackoverflow.com/questions/7171140/using-python-iterparse-for-large-xml-files
            element.clear(keep_tail=True)
            while element.getprevious() is not None:
                del element.getparent()[0]


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

    def save_intermediate(self, domain: str, target: str):
        output_dirpath = os.path.join(self.config.tmpdir, self.dataset_id, domain)
        target_path = os.path.join(output_dirpath, target.replace('.xml', '.arrow'))
        if os.path.isfile(target_path):
            return

        # todo
        schema = self.target_schema[target]
        schema_python = self.target_schema_python[target]

        # note, this drops values that arent int the schema
        def cast_dict_to_schema(schema, data: dict):
            d = dict()
            for k, type_ in schema.items():
                if k in data:
                    d[k] = type_(data[k])
                else:
                    d[k] = None
            return d

        with pa.OSFile(target_path, 'wb') as sink:
            with pa.ipc.new_file(sink, schema) as writer:
                xml_parse_stream = self.parse_xml(
                        os.path.join(output_dirpath, target), 
                        output_dirpath, 
                        )

                corrected_types_steam = map(
                        partial(cast_dict_to_schema, schema_python),
                        xml_parse_stream)

                chunked_stream = chunked(corrected_types_steam, self.batch_size)
                for chunk in tqdm(chunked_stream):
                    batch = pa.RecordBatch.from_pylist(
                            chunk,
                            schema=schema)

                    writer.write_batch(batch)

    def join_and_export(self, domain: str, disk_offload=True):
        # the max ram consumption is 2x the biggest table
        # because we want to sort the table fast with pyarrow
        # and it requires 2x because of immutability

        # save the tables to disk, to temporarily free the ram

        def save_table(table, path, batch_size):
            # saving it this way, because the higher level dataset api
            # doesn't keep the order during saving or loading
            print(f'saving {path}')
            with pa.OSFile(path, 'wb') as sink:
                with pa.ipc.new_file(sink, table.schema) as writer:
                    for batch in table.to_batches(max_chunksize=batch_size):
                        writer.write_batch(batch)

        def stream_rows_from_disk(path):
            #with pa.memory_map(os.path.join(output_dirpath, 'sorted_answers.arrow'), 'rb') as source:
            with pa.OSFile(path, 'rb') as source:
                source_stream = pa.ipc.open_file(source)
                for bi in range(source_stream.num_record_batches):
                    batch = source_stream.get_batch(bi)#.read_all()
                    for row in batch.to_pylist():
                        yield row

        def stream_rows(table):
            for batch in table.to_batches():
                for row in batch.to_pylist():
                    yield row

        output_dirpath = os.path.join(self.config.tmpdir, self.dataset_id, domain)
            
        print(f'joining users and posts of {domain}')
        posts_w_users_path = os.path.join(output_dirpath, 'Posts_w_Users.arrow')
        if not os.path.isfile(posts_w_users_path):
            postsds = ds.dataset(os.path.join(output_dirpath, 'Posts.arrow'), format='arrow')
            usersds = ds.dataset(os.path.join(output_dirpath, 'Users.arrow'), format='arrow')

            # note there are deleted users with a negative userid
            # todo
            # remove columns, because they crash pyarrow currently

            # this combination of dataset uses less ram than joining both as tables
            users_table = ds.dataset(usersds.to_table(columns=['Id', 'Reputation', 'Views', 'UpVotes', 'DownVotes', "AccountId", 'CreationDate']))
            print(f'joining posts_w_usrs {domain}')
            posts_w_users = postsds.join(users_table, keys='OwnerUserId', right_keys='Id', left_suffix='post', right_suffix='user').to_table(batch_size=self.batch_size)

            if disk_offload:
                save_table(posts_w_users, 
                        posts_w_users_path,
                        batch_size=self.batch_size)

            if disk_offload:
                del posts_w_users
            del users_table
            del usersds
            gc.collect()


        print('loading users and posts')
        posts_w_usersds = ds.dataset(posts_w_users_path, format='arrow')

        is_question = pc.field('PostTypeId') == 1
        is_answer = pc.field('PostTypeId') == 2

        # this specific way, we don't have to load the full postsds
        # the tables still run in memory
        # because pyarrow doesn't yet support sorting datasets on disk
        print('loading questions')
        question_path = os.path.join(output_dirpath, 'sorted_questions.arrow')
        if not os.path.isfile(question_path) or not disk_offload:
            question_table = posts_w_usersds.to_table(
                filter=is_question)

            print('sorting questions')
            sorted_questions = question_table.sort_by([('Id', 'ascending')])
            del question_table
            gc.collect()
            print('saving questions')

            if disk_offload:
                save_table(sorted_questions, question_path, self.batch_size)
                del sorted_questions
                gc.collect()

        print('loading answers')
        answer_path = os.path.join(output_dirpath, 'sorted_answers.arrow')
        if not os.path.isfile(answer_path) or not disk_offload:
            answer_table = posts_w_usersds.to_table(
                filter=is_answer)

            print('sorting answers')
            sorted_answers = answer_table.sort_by([('ParentId','ascending')])
            del answer_table
            gc.collect()
            print('saving answers')

            if disk_offload:
                save_table(sorted_answers, answer_path, self.batch_size)
                del sorted_answers
                gc.collect()

        if disk_offload:
            question_iter = stream_rows_from_disk(question_path)
            answer_iter = stream_rows_from_disk(answer_path)
        else:
            question_iter = stream_rows(sorted_questions)
            answer_iter = stream_rows(sorted_answers)

        a = next(answer_iter)
        appended = 0

        print(f'iterating: {domain}')
        for q in tqdm(question_iter):
            result = dict()
            result['question'] = q
            result['answers'] = []
            #print(q['Id'])
            while 1:
                assert q['Id'] <= a['ParentId'], f'question id not smaller than answer parent id, something broke, question: {q} \nanswer: {a} '
                if q['Id'] == a['ParentId']:
                    result['answers'] = result['answers'] + [a]
                    appended += 1
                    try:
                        a = next(answer_iter)
                    except StopIteration:
                        return
                else:
                    break

            #formatted = self.format(result)
            #self.save_to_jsonl(formatted)
            #print(appended)


class StackExchangeDataset(Dataset):
    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.scraper = StackExchangeScraper(self.config, self.id)
        self.processor = StackExchangeCodeProcessor(self.config, self.id)

    @property
    def info(self):
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


        return STACKEXCHANGEINFO

    @property
    def id(self):
        return "StackExchange"

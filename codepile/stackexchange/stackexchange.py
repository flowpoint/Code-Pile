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

from time import sleep

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
                'Comments.xml',
                #'PostHistory.xml',
                #'PostLinks.xml',
                'Posts.xml',
                #'Tags.xml',
                'Users.xml',
                #'Votes.xml',
                ]

        # key, python type, pyarrow type
        post_types = (
            ('Id', int, pa.int32),
            ('OwnerUserId', int, pa.int32),
            ('ParentId', int, pa.uint32),
            ('PostTypeId', int, pa.int8),
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
            ('UpVotes', int, pa.int32),
            ('DownVotes', int, pa.int32),
            ('AccountId', int, pa.int32),
            ('ProfileImageUrl', str, pa.large_string),

            ('CreationDate', datetime.fromisoformat, pa.date64),
            ('LastAccessDate', datetime.fromisoformat, pa.date64),
            )

        comment_types = (
            ('Id', int, pa.int32),
            ('PostId', int, pa.int32),
            ('Score', int, pa.int32), # can be negative
            ('Text', str, pa.large_string),
            ('UserId', int, pa.int32),
            ('CreationDate', datetime.fromisoformat, pa.date64),
            )

        post_schema = pa.schema([(k,a()) for k,p,a in post_types])
        post_schema_python = {k:p for k,p,a in post_types}

        user_schema = pa.schema([(k,a()) for k,p,a in user_types])
        user_schema_python = {k:p for k,p,a in user_types}

        comment_schema = pa.schema([(k,a()) for k,p,a in comment_types])
        comment_schema_python = {k:p for k,p,a in comment_types}

        self.target_schema_python = {
                'Users.xml': user_schema_python,
                'Posts.xml': post_schema_python,
                'Comments.xml': comment_schema_python
                }

        self.target_schema = {
                'Users.xml': user_schema,
                'Posts.xml': post_schema,
                'Comments.xml': comment_schema
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
                if domain != 'stackoverflow.com':
                    continue
                async_results[domain] = dict()
                for dump in dumps:
                    if parallel:
                        c = p.apply_async(self.extract_dumpfile, (domain, dump))
                        async_results[domain][dump] = c
                    else:
                        self.extract_dumpfile(domain, dump)

            # wait 
            for domain in g.keys():
                if domain != 'stackoverflow.com':
                    continue
                # wait until all domain targets are extracted
                for dump, result in async_results[domain].items():
                    result.wait()
                    assert result.successful()

        gc.collect()

        # transcode to arrow
        for domain in g.keys():
             # queue transcoding to intermediate format
            for target in self.target_files:
                if domain != 'stackoverflow.com':
                    continue
                self.save_intermediate(domain, target)
                gc.collect()

        # join tables to dicts
        for domain in g.keys():
            if domain != 'stackoverflow.com':
                continue
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

        def stream_batch_from_disk(path):
            #with pa.memory_map(os.path.join(output_dirpath, 'sorted_answers.arrow'), 'rb') as source:
            with pa.OSFile(path, 'rb') as source:
                source_stream = pa.ipc.open_file(source)
                for bi in range(source_stream.num_record_batches):
                    batch = source_stream.get_batch(bi)#.read_all()
                    for row in batch.to_pylist():
                        yield row

        def stream_rows(table):
            for batch in table.to_batches(max_chunksize=self.batch_size):
                for row in batch.to_pylist():
                    yield row

            

        '''
        # sort the users by their post
        split posts
        # answers already have their parentid
        join post_author with their postid
        join comment_authors with their commentid
        join comments with their postid

        exporting concept:
        the target is formatted as a thread document:

        iter questions
            add question_author 
            append comments
                add comment author
            append answers
                add aswer author
                append comments
                    add comment author

        therefore we know our key, beeing question['Id']
        we also know that:
        question['Id'] < answer['ParentId']

        we need the users alot, so for simplicity, keep the users table
        as a fast random access datastructure (dict) in memory

        we need comments two times, with separate iterators, 
        but both with sequential access

        it doesnt make a difference if we iteratively extend, 
        or group and aggregate the answers and comments

        '''


        output_dirpath = os.path.join(self.config.tmpdir, self.dataset_id, domain)

        '''
        def get_tag(output_dirpath):

            posts = postsds.to_table(columns=['Id', 'ParentId', 'OwnerUserId'])
            users = usersds.to_table(columns=['Id'])
            comments = commentsds.to_table(columns=['Id','PostId', 'UserId'])

            comments_authors = comments.join(users, keys='UserId', right_keys='Id',
                    left_suffix='comment', right_suffix='user')

            posts_authors = posts.join(users, keys='OwnerUserId', right_keys='Id',
                    left_suffix='post', right_suffix='user')

            posts_comments = posts_authors.join(comments_authors, keys='Id', right_keys='PostId',
                    left_suffix='post', right_suffix='comment')


            return posts_comments.sort_by('ParentId')

        tags = get_tag(output_dirpath)
        '''


        postsds = ds.dataset(os.path.join(output_dirpath, 'Posts.arrow'), format='arrow')
        usersds = ds.dataset(os.path.join(output_dirpath, 'Users.arrow'), format='arrow')
        commentsds = ds.dataset(os.path.join(output_dirpath, 'Comments.arrow'), format='arrow')

        sorted_questions_path = os.path.join(output_dirpath, 'sorted_Questions.arrow')
        sorted_answers_path = os.path.join(output_dirpath, 'sorted_Answers.arrow')
        sorted_comments_path = os.path.join(output_dirpath, 'sorted_Users.arrow')
        sorted_users_path = os.path.join(output_dirpath, 'sorted_Comments.arrow')

        ## 

        questions = postsds.to_table(filter=is_question)
        #sorted_questions = questions.sort_by('Id')
        save_table(sorted_questions, 
                    sorted_questions_path,
                    batch_size=self.batch_size)
        del questions
        del sorted_questions
        gc.collect()
        '''
        ##
        is_answer = pc.field('PostTypeId') == 2
        ##
        answers = postsds.to_table(filter=is_answer)
        ##
        '''
        sorted_answers = answers.sort_by('ParentId')
        save_table(sorted_answers, 
                    sorted_answers_path,
                    batch_size=self.batch_size)
        del answers
        del sorted_answers
        gc.collect()

        comments = commentsds.to_table()
        sorted_comments = comments.sort_by('PostId')
        save_table(sorted_comments, 
                    sorted_comments_path,
                    batch_size=self.batch_size)
        del comments
        del sorted_comments
        gc.collect()

        # keep users in memory, 
        # because of many random accesses
        # group to make lookup easier
        ##
        comments = commentsds.to_table()
        cols = set(comments.column_names)
        agg = list(zip(list(cols-{'PostId'}), itertools.cycle(['list']))) \
            + [('PostId','one')]
        grouped_comments = comments.group_by('PostId').aggregate(agg)

        del comments
        gc.collect()
        ##

        batch_size = 1000000

        partitioning = ds.partitioning(
                pa.schema([
                    #('PostTypeId', pa.int8()),
                    ('partition', pa.int64())
                    ])
                )
        ##
        ass = answers.add_column(len(answers.column_names)-1, pa.field('partition', pa.int64()), [[x//batch_size for x in range(0, answers.num_rows)]])
        ##
        is_question = pc.field('PostTypeId') == 1

        ##
        partitioned_posts = posts.add_column(len(posts.column_names)-1, 
                pa.field("partition", pa.int64()), 
                [[x//batch_size for x in range(posts.num_rows)]])

        ##
        ds.write_dataset(partitioned_posts, 'posts2.arrow', partitioning=partitioning, format='arrow')
        ##
        answers = ds.dataset('answers.arrow', format='arrow', partitioning=partitioning)
        qs = ds.dataset('questions.arrow', format='arrow', partitioning=partitioning)
        #users = usersds.to_table(columns=[j

        #qs = postsds.scanner(filter=is_question, batch_size=batch_size).to_batches()
        #qt = pa.Table.from_batches([next(qs)])

        ##
        ##
        q_partitions = qs.count_rows()//batch_size
        a_partitions = answers.count_rows()//batch_size
        ##
        postsds = ds.dataset('posts2.arrow', format='arrow', partitioning=partitioning)
        ##

        # join comments and users
        # inner join, or the comments would be uselessly replicated
        # 2 minutes
        batch_size=1000000
        joineds = []
        comments_parts = ceil(comments.num_rows/batch_size)

        for ci in tqdm(range(comments_parts)):
            #for ui in range(users_parts):
            ct = comments.slice(ci, batch_size)
            # join authors to comments
            ctj = ct.join(users, keys='UserId', right_keys='Id',
                    left_suffix='comment', right_suffix='user', 
                    join_type='inner')
            # join question partition to comments partition
            joineds.append(ctj)
            del ct
            del ctj
            gc.collect()
            break

        comments_w_users = pa.concat_tables(joineds)

        ##
        # join posts with users
        ##
        def generate_posts_w_users():
            batch_size=10000000
            joineds = []
            pi = ceil(postsds.count_rows()/batch_size)
            for p in tqdm(range(pi)):
                pt = postsds.to_table(filter=ds.field('partition') == p)
                ptj = pt.join(users, keys='OwnerUserId', right_keys='Id', 
                        left_suffix='post', right_suffix='user',
                        join_type='inner')
                ptjp = ptj.add_column(
                    len(ptj.column_names)-1, 
                    pa.field("partition", pa.int64()), 
                    [[p]*ptj.num_rows]
                    )
                #joineds.append(ptj)
                for b in ptj.to_batches():
                    yield b
                del pt
                del ptj
                gc.collect()


        partitioning = ds.partitioning(
                pa.schema([
                    #('PostTypeId', pa.int8()),
                    ('partition', pa.int64())
                    ])
                )

        #schem = next(iter(generate_posts_w_users()))
        ##
        ds.write_dataset(
                generate_posts_w_users(),
                #sc,
                os.path.join(output_dirpath, 'posts_w_users.arrow'),
                schema=schem.schema,
                format='arrow',
                partitioning=partitioning
                )

        ##
        posts_w_users = ds.dataset(os.path.join(output_dirpath, 'posts_w_users.arrow'),
                format='arrow',
                partitioning=partitioning)

        #posts_w_users = pa.concat_tables(joineds)
        ##

        #posts_w_users_w_comments = 
        joineds = []
        pp = ceil(posts_w_users.count_rows() / batch_size)
        for p in tqdm(range(pp)):
            ppt = posts_w_users.to_table(filter=ds.field('partition') == p)
            ptj = ppt.join(comments_w_users, keys='Id', right_keys='PostId', 
                    left_suffix='post', right_suffix='comment',
                    join_type='inner')
            #joineds.append(ptj)
            del ppt
            #del ptj
            gc.collect()

        #posts_w_users_w_comments = pa.concat_tables(joineds)
        ##
        questions = posts_w_users_w_comments.filter(is_question)
        ##
        answers = posts_w_users_w_comments.filter(is_answer)
        ##
        


        allq = []
        for qi in range(q_partitions):
            ##
            qt = qs.to_table(filter=ds.field('partition') == qi)

            joineds = []
            for c in commentsds.to_batches(batch_size=batch_size):
                ct = pa.Table.from_batches([c])
                # join authors to comments
                ctj = ct.join(usersds, keys='UserId', right_keys='Id', 
                        left_suffix='comment', right_suffix='comment')
                # join question partition to comments partition
                joined = qt.join(ctj, keys='Id', right_keys='PostId', 
                        left_suffix='question', right_suffix='comment', 
                        join_type='inner')
                joineds.append(joined)
                del ct
                del ctj
                gc.collect()

            return

            # combine comments partitions
            nt = pa.concat_tables(joineds)
            del joineds
            gc.collect()
            #snt = nt.sort_by('Idquestion')
            # join question authors to questions
            qc = nt.join(usersds, keys='OwnerUserId', right_keys='Id', 
                    left_suffix='question', right_suffix='author')


            for ai in range(a_partitions):
                at = answers.to_table(filter=ds.field('partition') == ai)

                joineds = []
                for c in commentsds.to_batches(batch_size=batch_size):
                    ct = pa.Table.from_batches([c])
                    ctj = ct.join(usersds, keys='UserId', right_keys='Id', 
                            left_suffix='comment', right_suffix='comment')
                    joined = at.join(ctj, keys='Id', right_keys='PostId', left_suffix='question', right_suffix='comment', join_type='inner')
                    joineds.append(joined)

                nt = pa.concat_tables(joineds)

                ac = nt.join(usersds, keys='OwnerUserId', right_keys='Id', 
                        left_suffix='question', right_suffix='author')
                    
            ##
            threads = qc.join(ac, keys='Id', right_keys='ParentId', 
                    left_suffix='question', right_suffix='answer')
                
            allq.append(threads)
                


            del qt
            del nt
            del ct
            del joined
            del joineds
            gc.collect()
        ##

        users = usersds.to_table()

        # now iteratively join all the tables
        iters = dict()

        iters['questions'] = stream_rows_from_disk(sorted_questions_path)
        #iters['question_comments'] = stream_rows_from_disk(sorted_comments_path)

        iters['answers'] = stream_rows_from_disk(sorted_answers_path)
        #iters['answer_comments'] = stream_rows_from_disk(sorted_comments_path)

        #a = next(answer_iter)
        #c = next(comment_iter)
        def get_user(user_id):
            user_pos = users['Id'].index(user_id).as_py()
            if user_pos != -1:
                 return users.take([user_pos]).to_pylist()[0]
            else:
                return {'unk'}

        def get_comments(post_id):
            comment_group_id = grouped_comments['PostId'].index(post_id).as_py()
            if comment_group_id != -1:
                 cols = grouped_comments.take([comment_group_id])
                 #return [cols[k][i] for k in cols.itercolumns
                 #print(cols.to_pylist())
                 cols.to_pydict()
                 return cols.to_pylist()
            else:
                return []

        for question in tqdm(iter['questions'])
            
        print(f'iterating: {domain}')
        # join question and answers iteratively, to save ram
        for question in tqdm(iters['questions']):
            # add author to the question
            question['Author'] = get_user(question['OwnerUserId'])

            # add the comments to the question
            question_comments = get_comments(
                    question['Id'])


            # add the authors to the question comments
            for i, c in enumerate(question_comments):
                print(c)
                question_comments[i]['Author'] = get_user(c['UserId_list'])


            # skip until the next related question answers
            itertools.takewhile(
                    lambda a: a['ParentId']<question['Id'],
                    iters['answers']
                )
            # collect the answers to the question
            answers = list(itertools.takewhile(
                    lambda a: a['ParentId']==question['Id'],
                    iters['answers']
                ))

            for i,a in enumerate(answers):
                # add authors to the answers
                answers[i]['Author'] = get_user(a['OwnerUserId'])

                # add the comments to the answers
                answer_comments = get_comments(
                        a['Id']
                        )
                # add authors to the comments of the answers
                for j, c in enumerate(answer_comments):
                    answer_comments[j]['Author'] = get_user(c['UserId_list'])

            thread = dict()
            thread['question'] = question
            thread['answers'] = answers
            print(thread)
            sleep(2)
            '''


class StackExchangeDataset(Dataset):
    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.scraper = StackExchangeScraper(self.config, self.id)
        self.processor = StackExchangeCodeProcessor(self.config, self.id)

    @property
    def info(self):
        # todo
        description = '''a question answer dataset of all stackexchange sites.
        every sample consists of a question, a list of answers and users tht asked or answered, with additional metadata.
        the data is obtained through the stackoverflow internet archive dump.

        for more information about the dump, see the dumps data schema here:
        
        https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede
        '''


        # todo

        STACKEXCHANGEINFO = DatasetInfo(
                id='StackExchange',
                description=description,
                data_end=datetime(2022,6,6),
                data_start=datetime(2014,1,21),
                size=-1,
                storage_format='.jsonl.zst',
                #storage_uri='/root',
                cpu_hours=-1,
                gpu_hours=0,
                # 128GB in bits
                ram_requirements=1024000000000,
                # 550GB in bits disk
                tempfile_requirement=4400000000000,
                source_uri='https://archive.org/details/stackexchange',
                dataset_pros='',
                dataset_cons='',
                languages=[''],
                coding_languages=[''],
                modalities=['discussion'],
                source_license='CC BY-SA 4.0',
                source_citation='',
                data_owner='flowpoint',
                contributers=['vangap']
                )

        return STACKEXCHANGEINFO

    @property
    def id(self):
        return "StackExchange"

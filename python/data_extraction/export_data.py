from data_extraction import content_export
from data_extraction import taxonomy_query
from lib import plek
from lib.helpers import dig
from multiprocessing import Pool
from git import Repo


import datetime
import data
import yaml
import functools
import progressbar
import gzip
import json
import os
import ijson


config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'config', 'data_export_fields.json')

with open(config_path) as json_data_file:
    configuration = json.load(json_data_file)


def export_content(output_filename="data/content.json.gz"):
    seen_content_ids = set()
    duplicate_content_ids = []

    def filter_content(content):
        # This can happen a few ways, for example, if the request to
        # get the content resulted in a redirect.
        if not content:
            return False

        content_id = content['content_id']

        if content_id in seen_content_ids:
            duplicate_content_ids.append(content_id)
            return False

        seen_content_ids.add(content_id)
        return True

    content_iterator, count = __get_all_content()
    content = filter(filter_content, content_iterator)

    progress_bar = progressbar.ProgressBar(max_value=count)

    with gzip.open(output_filename, 'wt') as output_file:
        __stream_json(output_file, progress_bar(content))

    duplicate_content_ids_count = len(set(duplicate_content_ids))
    print("Seen {} duplicate content ids".format(
        duplicate_content_ids_count
    ))


def export_filtered_content(input_filename="data/content.json.gz", output_filename="data/filtered_content.json.gz"):
    slicer = functools.partial(content_export.content_dict_slicer,
                               base_fields=configuration['base_fields'],
                               taxon_fields=configuration['taxon_fields'],
                               ppo_fields=configuration['ppo_fields'])

    __transform_content(input_filename=input_filename,
                        output_filename=output_filename,
                        transform_function=lambda iterator: map(slicer, iterator))


def export_untagged_content(input_filename="data/content.json.gz", output_filename="data/untagged_content.json.gz"):
    def __filter_tagged(dict_in):
        return dig(dict_in, 'links', 'taxons') is None

    untagged_dict_slicer = functools.partial(content_export.untagged_dict_slicer,
                                             base_fields=configuration['untagged_content_fields'],
                                             ppo_fields=configuration['ppo_fields'])

    __transform_content(input_filename=input_filename,
                        output_filename=output_filename,
                        transform_function=lambda iterator: map(untagged_dict_slicer, filter(__filter_tagged, iterator)))


def export_taxons(output_filename="data/taxons.json.gz"):
    def __taxonomy():
        query = taxonomy_query.TaxonomyQuery()
        level_one_taxons = query.level_one_taxons()
        children = [taxon
                    for level_one_taxon in level_one_taxons
                    for taxon in query.child_taxons(level_one_taxon['base_path'])]
        return iter(level_one_taxons + children)

    with gzip.open(output_filename, 'wt') as output_file:
        __stream_json(output_file, __taxonomy())


def __stream_json(output_file, iterator):
    # The json package in the stdlib doesn't support dumping a
    # generator, but it can handle lists, so this class acts as a
    # go between, making the generator look like a list.
    class StreamContent(list):
        def __init__(self, iterator):
            self.iterator = iterator

        def __bool__(self):
            # The json class tests the truthyness of this object,
            # so this needs to be overridden to True
            return True

        def __iter__(self):
            return self.iterator

    obj = {"_meta": {"date": str(datetime.datetime.today()),
                     "code": Repo(search_parent_directories=True).head.commit.hexsha},
           "items": StreamContent(iterator)}
    
    json.dump(
        obj,
        output_file,
        indent=4,
        check_circular=False,
        sort_keys=True,
    )


def __transform_content(input_filename="data/content.json.gz",
                        output_filename="data/filtered_content.json.gz",
                        transform_function=lambda x: x):
    with gzip.open(input_filename, mode='rt') as input_file:
        with gzip.open(output_filename, mode='wt') as output_file:
            content_generator = ijson.items(input_file, prefix='item')
            __stream_json(output_file, transform_function(content_generator))


def __get_all_content():
    get_content = functools.partial(
        content_export.get_content,
        content_store_url=plek.find('draft-content-store')
    )

    blacklisted_document_types = data.document_types_excluded_from_the_topic_taxonomy()
    progress_bar = progressbar.ProgressBar()

    content_links_list = list(
        progress_bar(
            content_export.content_links_generator(
                blacklist_document_types=blacklisted_document_types
            )
        )
    )

    content_links_set = set(content_links_list)
    duplicate_links = len(content_links_list) - len(content_links_set)

    if duplicate_links > 0:
        print("{} duplicate links from Rummager".format(duplicate_links))

    pool = Pool(10)
    return pool.imap(get_content, content_links_set), len(content_links_set)

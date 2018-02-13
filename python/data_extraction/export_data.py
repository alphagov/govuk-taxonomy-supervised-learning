from data_extraction import content_export
from data_extraction import taxonomy_query
from lib import json_arrays, plek
from lib.helpers import dig
import functools
from multiprocessing import Pool
import gzip
import json
import os


config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'config', 'data_export_fields.json')

with open(config_path) as json_data_file:
    configuration = json.load(json_data_file)


def __transform_content(input_filename="data/content.json.gz",
                        output_filename="data/filtered_content.json.gz",
                        transform_function=lambda x: x):
    with gzip.open(input_filename, mode='rt') as input_file:
        with gzip.open(output_filename, mode='wt') as output_file:
            content_generator = json_arrays.read_json(input_file)
            json_arrays.write_json(output_file, transform_function(content_generator))


def export_content(output_filename="data/content.json.gz"):
    def __complete_content():
        get_content = functools.partial(content_export.get_content,
                                        content_store_url=plek.find('draft-content-store'))

        content_links_generator = content_export.content_links_generator(
            blacklist_document_types=configuration['blacklist_document_types']
        )

        pool = Pool(10)
        return pool.imap(get_content, content_links_generator)

    with gzip.open(output_filename, 'wt') as output_file:
        json_arrays.write_json(output_file,
                               filter(lambda link: link, __complete_content()))


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
        json_arrays.write_json(output_file, __taxonomy())

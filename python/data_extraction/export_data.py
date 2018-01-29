from data_extraction import content_export
from data_extraction import taxonomy_query, plek
import json
import sys
import functools
from multiprocessing import Pool


with open('config/data_export_fields.json') as json_data_file:
    configuration = json.load(json_data_file)

content_links_generator = content_export.content_links_generator(
    blacklist_document_types=configuration['blacklist_document_types']
)

get_content = functools.partial(content_export.get_content,
                                base_fields=configuration['base_fields'],
                                taxon_fields=configuration['taxon_fields'],
                                ppo_fields=configuration['ppo_fields'],
                                content_store_url=plek.find('draft-content-store'))


def taxonomy():
    query = taxonomy_query.TaxonomyQuery()
    level_one_taxons = query.level_one_taxons()
    children = [taxon
                for level_one_taxon in level_one_taxons
                for taxon in query.child_taxons(level_one_taxon['base_path'])]
    return iter(level_one_taxons + children)


def content():
    pool = Pool(10)
    content_generator = pool.imap(get_content, content_links_generator, 50)
    return filter(lambda link: link, content_generator)



def export_content(output="data/content.json"):
    __write_json(output, content())


def export_taxons(output="data/taxons.json"):
    __write_json(output, taxonomy())


# PRIVATE

def __write_json(filename, generator):
    with open(filename, 'w') as file:
        file.write("[ ")
        file.write(json.dumps(next(generator)))
        for index, taxon in enumerate(generator):
            file.write(",\n")
            file.write(json.dumps(taxon))
            if index % 1000 is 0:
                print("Documents exported: %s" % index)
        file.write("]\n")

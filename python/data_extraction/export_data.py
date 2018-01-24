from data_extraction import content_export
from data_extraction import taxonomy_query, plek
import json
import functools

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
    content_generator = map(get_content, content_links_generator)
    return filter(lambda link: link, content_generator)


def export_content():
    __write_json("/tmp/content.json", content())


def export_taxons():
    __write_json("/tmp/taxons.json", taxonomy())


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

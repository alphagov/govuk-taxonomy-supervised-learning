import gzip
from lib import json_arrays
from lib.helpers import dig
from lib import services
from statistics import mean
from data_extraction.taxonomy_query import TaxonomyQuery


def measure_average_taxons(filename):

    __query = TaxonomyQuery()

    def __number_of_taxons(content_item):
        taxons = dig(content_item, 'links', 'taxons') or []
        return len([taxon for taxon in taxons if __query.taxon_linked_to_root(taxon)])

    with gzip.open(filename, mode='rt') as file:
        content_generator = json_arrays.read_json(file)
        value = mean(map(__number_of_taxons, filter(__query.content_linked_to_root, content_generator)))
        services.statsd.gauge('average_taxons_per_content_item', value)

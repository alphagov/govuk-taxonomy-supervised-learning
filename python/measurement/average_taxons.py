import gzip
import ijson
import progressbar
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

    progress_bar = progressbar.ProgressBar()

    with gzip.open(filename, mode='rt') as file:
        content_items = ijson.items(file, prefix='item')

        value = mean(
            filter(
                # Don't count content not tagged to the topic
                # taxonomy, as removing this gives a more useful
                # measure of what the makeup of the tagged content is
                lambda count: count != 0,
                map(__number_of_taxons, progress_bar(content_items))
            )
        )

        print("Found an average of {} taxons per content item".format(value))
        services.statsd.gauge('average_taxons_per_content_item', value)

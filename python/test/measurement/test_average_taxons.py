import unittest
import unittest.mock
from test.data_extraction import content_store_helpers as helpers
from measurement import average_taxons
import json
import io


class TestAverageTaxons(unittest.TestCase):
    def test_measure_average_taxons(self):
        input_string = "[" + json.dumps(helpers.content_with_multiple_taxons) + ",\n" + \
                             json.dumps(helpers.content_with_multiple_taxons) + ",\n" + \
                             json.dumps(helpers.content_with_multiple_unconnected_taxons) + "\n]"
        with unittest.mock.patch('gzip.open', return_value=io.StringIO(input_string)):
            with unittest.mock.patch('lib.services.statsd') as statsd_mock:
                average_taxons.measure_average_taxons('')
                statsd_mock.gauge.assert_called_once_with('average_taxons_per_content_item', 1)

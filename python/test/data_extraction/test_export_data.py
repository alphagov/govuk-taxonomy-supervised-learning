import unittest
import unittest.mock
from data_extraction import export_data
from test.data_extraction.content_store_helpers import *
import responses
import json
import io
from test.lib.mock_io import MockIO

class TestExportData(unittest.TestCase):
    @responses.activate
    def test_level_export_taxons(self):
        content_store_has_item('/taxons/root_taxon', multi_level_child_taxons)
        content_store_has_item("/", json=level_one_taxons)

        output = MockIO()
        with unittest.mock.patch('gzip.open', return_value=output):
            export_data.export_taxons()
            expected = [{"base_path": "/taxons/root_taxon", "content_id": "rrrr"},
                        {"parent_content_id": "rrrr", "base_path": "/root_taxon/taxon_a", "content_id": "aaaa"},
                        {"parent_content_id": "aaaa", "base_path": "/root_taxon/taxon_1", "content_id": "aaaa_1111"},
                        {"parent_content_id": "aaaa", "base_path": "/root_taxon/taxon_2", "content_id": "aaaa_2222"}]
            self.assertEqual(expected, json.loads(output.buffer))

    def return_input_or_output(self, input_io, output_io):
        return lambda *_, **ka: input_io if ka['mode'] == 'rt' else output_io

    @responses.activate
    def test_export_content(self):
        output = MockIO()
        responses.add(
            responses.GET,
            "{}/search.json".format(plek.find("rummager")),
            json=content_links
        )
        content_store_has_item(content_first['base_path'], json=content_first)
        content_store_has_item(content_second['base_path'], json=content_second)
        with unittest.mock.patch('gzip.open', return_value=output):
            export_data.export_content()
            expected = [content_first, content_second]
            self.assertCountEqual(expected, json.loads(output.buffer))

    def test_export_filtered_content(self):
        input_string = "[" + json.dumps(content_with_taxons) + ",\n" + json.dumps(content_without_taxons) + "\n]"
        output = MockIO()
        with unittest.mock.patch('gzip.open',
                                 side_effect=self.return_input_or_output(io.StringIO(input_string), output)):
            export_data.export_filtered_content()
            expected = [{"taxons": content_with_taxons['links']['taxons'],
                         "base_path": content_with_taxons['base_path'],
                         "content_id": content_with_taxons["content_id"]},
                        {"base_path": content_without_taxons['base_path'],
                         "content_id": content_without_taxons["content_id"]}]
            self.assertEqual(expected, json.loads(output.buffer))

    def export_untagged_content(self):
        output = MockIO()
        input_string = "[" + json.dumps(content_with_taxons) + ",\n" + json.dumps(content_without_taxons) + "\n]"

        with unittest.mock.patch('gzip.open',
                                 side_effect=self.return_input_or_output(io.StringIO(input_string), output)):
            export_data.export_filtered_content()
            self.assertListEqual([content_without_taxons], json.loads(output.buffer))

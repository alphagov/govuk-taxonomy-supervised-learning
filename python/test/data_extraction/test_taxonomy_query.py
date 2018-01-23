import unittest
from data_extraction import taxonomy_query
from data_extraction import plek
import responses


class TestTaxonomyQuery(unittest.TestCase):

    @responses.activate
    def test_level_one_taxons(self):
        content_store_has_item("/",
                               json={"links": {"level_one_taxons": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]}})
        self.assertCountEqual(taxonomy_query.TaxonomyQuery(["a"]).level_one_taxons(), [{"a": 1}, {"a": 3}])


def content_store_has_item(path, json):
    responses.add(responses.GET, plek.find("draft-content-store") + "/content" + path,
                  json=json, status=200)

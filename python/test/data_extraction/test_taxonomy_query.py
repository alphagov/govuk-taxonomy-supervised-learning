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

    @responses.activate
    def test_child_taxons_empty_array(self):
        content_store_has_item('/taxons/root_taxon', no_taxons())
        self.assertListEqual(taxonomy_query.TaxonomyQuery().child_taxons('/taxons/root_taxon'), [])

    @responses.activate
    def test_child_taxons_single_level(self):
        content_store_has_item('/taxons/root_taxon', single_level_child_taxons('rrrr', 'aaaa', 'bbbb'))
        taxons = taxonomy_query.TaxonomyQuery().child_taxons('/taxons/root_taxon')
        expected = [{'content_id': 'aaaa', 'base_path': '/taxons/aaaa', 'parent_content_id': 'rrrr'},
                    {'content_id': 'bbbb', 'base_path': '/taxons/bbbb', 'parent_content_id': 'rrrr'}]
        self.assertCountEqual(taxons, expected)

    @responses.activate
    def test_child_taxons_multiple_levels(self):
        content_store_has_item('/taxons/root_taxon', multi_level_child_taxons())
        taxons = taxonomy_query.TaxonomyQuery().child_taxons('/taxons/root_taxon')
        expected = [{'content_id': 'aaaa', 'base_path': '/root_taxon/taxon_a', 'parent_content_id': 'rrrr'},
                    {'content_id': 'aaaa_1111', 'base_path': '/root_taxon/taxon_1', 'parent_content_id': 'aaaa'},
                    {'content_id': 'aaaa_2222', 'base_path': '/root_taxon/taxon_2', 'parent_content_id': 'aaaa'}]
        self.assertCountEqual(taxons, expected)


def content_store_has_item(path, json):
    responses.add(responses.GET, plek.find("draft-content-store") + "/content" + path,
                  json=json, status=200)


def multi_level_child_taxons():
    return {
        "base_path": "/taxons/root_taxon",
        "content_id": "rrrr",
        "links": {
            "child_taxons": [
                {
                    "base_path": "/root_taxon/taxon_a",
                    "content_id": "aaaa",
                    "links": {
                        "child_taxons": [
                            {
                                "base_path": "/root_taxon/taxon_1",
                                "content_id": "aaaa_1111",
                                "links": {}
                            },
                            {
                                "base_path": "/root_taxon/taxon_2",
                                "content_id": "aaaa_2222",
                                "links": {}
                            }
                        ]
                    }
                }
            ]
        }
    }


def single_level_child_taxons(root, child_1, child_2):
    return {
        "base_path": "/taxons/#{root}",
        "content_id": root,
        "links": {
            "child_taxons": [
                {
                    "base_path": "/taxons/%s" % child_1,
                    "content_id": child_1,
                    "links": {}
                },
                {
                    "base_path": "/taxons/%s" % child_2,
                    "content_id": child_2,
                    "links": {}
                }
            ]
        }
    }


def no_taxons():
    return {
        "base_path": "/",
        "content_id": "f3bbdec2-0e62-4520-a7fd-6ffd5d36e03a"
    }

import unittest
from data_extraction import taxonomy_query
from test.data_extraction import content_store_helpers as helpers
import responses


class TestTaxonomyQuery(unittest.TestCase):

    @responses.activate
    def test_level_one_taxons(self):
        helpers.content_store_has_item("/",
                               json={"links": {"level_one_taxons": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]}})
        self.assertCountEqual(taxonomy_query.TaxonomyQuery(["a"]).level_one_taxons(), [{"a": 1}, {"a": 3}])

    @responses.activate
    def test_child_taxons_empty_array(self):
        helpers.content_store_has_item('/taxons/root_taxon', helpers.content_without_taxons)
        self.assertListEqual(taxonomy_query.TaxonomyQuery().child_taxons('/taxons/root_taxon'), [])

    @responses.activate
    def test_child_taxons_single_level(self):
        helpers.content_store_has_item('/taxons/root_taxon', helpers.single_level_child_taxons('rrrr', 'aaaa', 'bbbb'))
        taxons = taxonomy_query.TaxonomyQuery().child_taxons('/taxons/root_taxon')
        expected = [{'content_id': 'aaaa', 'base_path': '/taxons/aaaa', 'parent_content_id': 'rrrr'},
                    {'content_id': 'bbbb', 'base_path': '/taxons/bbbb', 'parent_content_id': 'rrrr'}]
        self.assertCountEqual(taxons, expected)

    @responses.activate
    def test_child_taxons_multiple_levels(self):
        helpers.content_store_has_item('/taxons/root_taxon', helpers.multi_level_child_taxons)
        taxons = taxonomy_query.TaxonomyQuery().child_taxons('/taxons/root_taxon')
        expected = [{'content_id': 'aaaa', 'base_path': '/root_taxon/taxon_a', 'parent_content_id': 'rrrr'},
                    {'content_id': 'aaaa_1111', 'base_path': '/root_taxon/taxon_1', 'parent_content_id': 'aaaa'},
                    {'content_id': 'aaaa_2222', 'base_path': '/root_taxon/taxon_2', 'parent_content_id': 'aaaa'}]
        self.assertCountEqual(taxons, expected)


class ContentLinkedToRoot(unittest.TestCase):

    def __connected(self, dict):
        return taxonomy_query.TaxonomyQuery().content_linked_to_root(dict)

    def test_no_taxons(self):
        self.assertFalse(self.__connected(helpers.content_without_taxons))

    def test_unconnected_taxon(self):
        self.assertFalse(self.__connected(helpers.content_with_multiple_unconnected_taxons))

    def test_directly_connected_taxon(self):
        self.assertTrue(self.__connected(helpers.content_with_direct_link_to_root_taxon))

    def test_indirectly_connected_multiple_taxons(self):
        self.assertTrue(self.__connected(helpers.content_with_multiple_taxons))


class TaxonLinkedToRoot(unittest.TestCase):
    def __connected(self, dict):
        return taxonomy_query.TaxonomyQuery().taxon_linked_to_root(dict)

    def test_linked_to_root(self):
        self.assertFalse(self.__connected(helpers.content_with_multiple_taxons['links']['taxons'][0]))
        self.assertTrue(self.__connected(helpers.content_with_multiple_taxons['links']['taxons'][1]))


import unittest

from data.taxons import *

TAXON_A = {
    "content_id": "A",
    "base_path": "a",
    "links": {}
}
TAXON_ROOT = {
    "content_id": "ROOT",
    "base_path": "/"
}
TAXON_B = {
    "content_id": "B",
    "base_path": "b",
    "links": {
        "root_taxon": [TAXON_ROOT]
    }
}
TAXON_C = {
    "content_id": "C",
    "links": {
        "parent_taxons": [TAXON_B]
    }
}

class TestTaxons(unittest.TestCase):
    def test_content_item_taxons(self):
        self.assertListEqual(
            list(
                content_item_taxons({ "links": [] })
            ),
            []
        )

        self.assertListEqual(
            list(
                content_item_taxons({
                    "links": {
                        "taxons": [TAXON_A]
                    }
                })
            ),
            [TAXON_A]
        )

        self.assertListEqual(
            list(
                content_item_taxons({
                    "links": {
                        "taxons": [TAXON_A, TAXON_C]
                    }
                })
            ),
            [TAXON_A, TAXON_C, TAXON_B, TAXON_ROOT]
        )

    def test_content_item_tagged_to_topic_taxonomy(self):
        self.assertTrue(
            content_item_tagged_to_topic_taxonomy({
                "links": {
                    "taxons": [TAXON_B]
                }
            })
        )

        self.assertFalse(
            content_item_tagged_to_topic_taxonomy({
                "links": {}
            })
        )

        self.assertFalse(
            content_item_tagged_to_topic_taxonomy({
                "links": {
                    "taxons": [TAXON_A]
                }
            })
        )

    def test_content_item_within_part_of_taxonomy(self):
        self.assertTrue(
            content_item_within_part_of_taxonomy(
                {
                    "links": {
                        "taxons": [TAXON_C]
                    }
                },
                taxonomy_part_content_id="B"
            )
        )

        self.assertFalse(
            content_item_within_part_of_taxonomy(
                {
                    "links": {
                        "taxons": [TAXON_A]
                    }
                },
                taxonomy_part_content_id="B"
            )
        )

    def test_content_item_directly_tagged_to_taxon(self):
        self.assertTrue(
            content_item_directly_tagged_to_taxon(
                {
                    "links": {
                        "taxons": [TAXON_A]
                    }
                },
                taxon_content_id="A"
            )
        )

        self.assertFalse(
            content_item_directly_tagged_to_taxon(
                {
                    "links": {
                        "taxons": [TAXON_C]
                    }
                },
                taxon_content_id="B"
            )
        )


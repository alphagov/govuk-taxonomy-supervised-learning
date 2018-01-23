import unittest
from data_extraction import content_export
import responses


def content_links():
    return {
        'results': [{'link': '/first/path'}, {'link': '/second/path'}]
    }


class ContentLinksGenerator(unittest.TestCase):

    @responses.activate
    def test_content_links_generator(self):
        responses.add(responses.GET, "http://example.com/search.json", json=content_links())
        response = content_export.content_links_generator(rummager_url="http://example.com")
        self.assertListEqual(list(response), ['/first/path', '/second/path'])

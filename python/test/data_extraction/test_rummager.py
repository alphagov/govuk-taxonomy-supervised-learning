import unittest
from data_extraction import rummager
import responses


def stub_rummager(json, start, count):
    responses.add(responses.GET, "http://example.com/search.json?start={start}&count={count}".format(start=start, count=count),
                  json=json, status=200, match_querystring=True)


class TestRummager(unittest.TestCase):

    @responses.activate
    def test_search_generator(self):
        stub_rummager({'results': [{'title': 't1'}, {'title': 't2'}]}, 0, 2)
        stub_rummager({'results': [{'title': 't3'}]}, 2, 2)

        search_results = [{'title': 't1'}, {'title': 't2'}, {'title': 't3'}]
        results = rummager.Rummager("http://example.com").search_generator({}, 2)
        self.assertListEqual(search_results, list(results))

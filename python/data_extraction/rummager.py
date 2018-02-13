from itertools import count
from lib.helpers import merge
from lib import plek
import requests


class Rummager():

    def __init__(self, base_url=plek.find("rummager")):
        self.base_url = base_url

    def search_generator(self, args, page_size=100):
        for index in count(0, page_size):
            search_params = merge(args, {"start": index, "count": page_size})
            results = self.__search(search_params).get('results', [])
            for result in results:
                yield result
            if len(results) < page_size:
                break

    # PRIVATE

    def __search(self, args):
        request_url = "{base_url}/search.json".format(base_url=self.base_url)
        return requests.get(request_url, args).json()

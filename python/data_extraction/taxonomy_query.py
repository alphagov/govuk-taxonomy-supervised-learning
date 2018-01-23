from data_extraction import plek
import requests
from data_extraction.helpers import slice, dig


class TaxonomyQuery():
    def __init__(self, key_list=("content_id", "base_path", "title"),
                 content_store_url=plek.find("draft-content-store")):
        self.content_store_url = content_store_url
        self.key_list = key_list

    def level_one_taxons(self):
        taxons = dig(self.__get_content_hash('/'), "links", "level_one_taxons")
        return [slice(taxon, self.key_list) for taxon in taxons]

    # PRIVATE

    def __get_content_hash(self, path):
        url = "{base}/content{path}".format(base=self.content_store_url, path=path)
        return requests.get(url).json()

from lib import plek
import requests
from lib.helpers import slice, dig


class TaxonomyQuery():
    def __init__(self, key_list=("content_id", "base_path", "title"),
                 content_store_url=plek.find("draft-content-store")):
        self.content_store_url = content_store_url
        self.key_list = key_list

    def level_one_taxons(self):
        taxons = dig(self.__get_content_dict('/'), "links", "level_one_taxons")
        return [slice(taxon, self.key_list) for taxon in taxons]

    def child_taxons(self, base_path):
        root_content_dict = self.__get_content_dict(base_path)
        taxons = dig(root_content_dict, "links", "child_taxons") or []
        return self.__recursive_child_taxons(taxons, root_content_dict['content_id'])

    # PRIVATE

    def __build_child_dict(self, taxon, parent_content_id):
        sliced_dict = slice(taxon, key_list=self.key_list)
        sliced_dict['parent_content_id'] = parent_content_id
        return sliced_dict

    @staticmethod
    def __child_taxons(taxon):
        return dig(taxon, 'links', 'child_taxons') or []

    def __recursive_child_taxons(self, taxons, parent_content_id):
        current_taxons = [self.__build_child_dict(taxon, parent_content_id) for taxon in taxons]
        children = [descendents
                    for taxon in taxons
                    for descendents in self.__recursive_child_taxons(self.__child_taxons(taxon), taxon['content_id'])
                    ]
        return current_taxons + children

    def __get_content_dict(self, path):
        url = "{base}/content{path}".format(base=self.content_store_url, path=path)
        return requests.get(url).json()

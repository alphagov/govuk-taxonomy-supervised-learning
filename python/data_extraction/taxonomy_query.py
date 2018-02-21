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

    def taxon_linked_to_root(self, dict):
        if dict is None:
            return False
        links = dict.get("links")
        if "root_taxon" in links:
            return True
        return self.taxon_linked_to_root(dig(links, "parent_taxons", 0))

    def content_linked_to_root(self, content_dict):
        if dig(content_dict, "links", "root_taxon") is not None:
            return True
        taxons = dig(content_dict, "links", "taxons") or []
        return any(self.taxon_linked_to_root(taxon) for taxon in taxons)

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

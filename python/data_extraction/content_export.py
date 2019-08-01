from data_extraction import rummager
from lib import plek
from lib.helpers import slice, dig, merge
import requests


def content_links_generator(page_size=1000,
                            additional_search_fields = {},
                            rummager_url=plek.find('search-api'),
                            blacklist_document_types=[]):
    search_dict = merge({'reject_content_store_document_type': blacklist_document_types,
                         'fields': ['link'],
                         'debug': 'include_withdrawn'},
                        additional_search_fields)
    search_results = rummager.Rummager(rummager_url).search_generator(search_dict, page_size=page_size)
    return map(lambda h: h.get('link'), search_results)


def content_dict_slicer(content_dict, base_fields=[], taxon_fields=[], ppo_fields=[]):
    result = slice(content_dict, base_fields)
    taxons = dig(content_dict, 'links', 'taxons')
    ppo = dig(content_dict, 'links', 'primary_publishing_organisation')

    if taxons:
        result['taxons'] = [slice(taxon, taxon_fields) for taxon in taxons]
    if ppo:
        result['primary_publishing_organisation'] = slice(ppo[0], ppo_fields)

    return result


def untagged_dict_slicer(content_dict, base_fields=[], ppo_fields=[]):
    result = slice(content_dict, base_fields)

    logo = dig(content_dict, 'links', 'organisations', 0, 'details', 'logo', 'formatted-title')
    ppo = dig(content_dict, 'links', 'primary_publishing_organisation')

    if logo:
        result['logo'] = logo
    if ppo:
        result['primary_publishing_organisation'] = slice(ppo[0], ppo_fields)

    return result


def get_content(base_path,
                content_store_url=plek.find('content-store')):
    content_dict = __get_content_dict(base_path, content_store_url)

    if not content_dict:
        return False

    # Skip this if we don't get back the content we expect, e.g. if
    # the Content Store has redirected the request
    if content_dict.get('base_path') != base_path:
        return False

    # Skip anything without a content_id
    if 'content_id' not in content_dict:
        return False

    return content_dict


# PRIVATE


def __get_content_dict(path, content_store_url):

    url = "{content_store_url}/content{path}".format(content_store_url=content_store_url, path=path)
    response = requests.get(url)
    if response.status_code == 200:
        return requests.get(url).json()
    else:
        return False

from data_extraction import rummager
from data_extraction import plek
from data_extraction.helpers import slice, dig
import requests

def content_links_generator(page_size=1000, rummager_url=plek.find('rummager'), blacklist_document_types=[]):
    search_results = rummager.Rummager(rummager_url).search_generator(
        {'reject_content_store_document_type': blacklist_document_types, 'fields': ['link']}, page_size=page_size)
    return map(lambda h: h.get('link'), search_results)


def content_hash_slicer(content_hash, base_fields=[], taxon_fields=[], ppo_fields=[]):
    result = slice(content_hash, base_fields)
    taxons = dig(content_hash, 'links', 'taxons')
    ppo = dig(content_hash, 'links', 'primary_publishing_organisation')

    if taxons:
        result['taxons'] = [slice(taxon, taxon_fields) for taxon in taxons]
    if ppo:
        result['primary_publishing_organisation'] = slice(ppo[0], ppo_fields)

    return result


def get_content(base_path,
                content_store_url=plek.find('draft-content-store')):
    content_hash = __get_content_hash(base_path, content_store_url)

    # Skip this if we don't get back the content we expect, e.g. if
    # the Content Store has redirected the request
    if content_hash.get('base_path') != base_path:
        return {}

    # Skip anything without a content_id
    if 'content_id' not in content_hash:
        return {}

    return content_hash


# PRIVATE


def __get_content_hash(path, content_store_url):

    url = "{content_store_url}/content{path}".format(content_store_url=content_store_url, path=path)
    response = requests.get(url)
    if response.status_code == 200:
        return requests.get(url).json()
    else:
        return {}

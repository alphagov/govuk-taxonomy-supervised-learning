from data_extraction import rummager
from data_extraction import plek
from data_extraction.helpers import slice, dig
import requests

def content_links_generator(page_size=1000, rummager_url=plek.find('rummager'), blacklist_document_types=[]):
    search_results = rummager.Rummager(rummager_url).search_generator(
        {'reject_content_store_document_type': blacklist_document_types, 'fields': ['link']}, page_size=page_size)
    return map(lambda h: h.get('link'), search_results)

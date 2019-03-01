"""Extract content from from data/content.json
"""
# coding: utf-8

import os
import csv
import logging.config
from itertools import islice

from data_extraction.export_data import jenkins_compatible_progress_bar
from pipeline_functions import extract_text, get_primary_publishing_org, get_text, map_content_id_to_taxon_id
from tokenizing import create_and_save_tokenizer
import yaml
from data import *



# Setup pipeline logging

LOGGING_CONFIG = os.getenv('LOGGING_CONFIG')
logging.config.fileConfig(LOGGING_CONFIG)
logger = logging.getLogger('clean_content_links')

DATADIR = os.getenv('DATADIR')

OUTPUT_CLEAN_CONTENT = os.path.join(DATADIR, 'clean_content_links.csv.temp')
OUTPUT_CONTENT_TO_TAXON_MAP = os.path.join(DATADIR, 'content_to_taxon_map_links.csv.temp')

OUTPUT_TEXT_TOKENIZER = os.path.join(DATADIR, 'combined_text_tokenizer_links.json.temp')
OUTPUT_TITLE_TOKENIZER = os.path.join(DATADIR, 'title_tokenizer_links.json.temp')
OUTPUT_DESCRIPTION_TOKENIZER = os.path.join(DATADIR, 'description_tokenizer_links.json.temp')

OUTPUT_METADATA_LISTS = os.path.join(DATADIR, 'metadata_lists_links.yaml.temp')


class Metadata:
    def __init__(self):
        self.document_types = set()
        self.primary_publishing_organisations = set()
        self.publishing_apps = set()

    def write(self):
        logger.info('saving metadata lists')

        with open(OUTPUT_METADATA_LISTS, "w") as f:
            yaml.dump({
                'document_type': sorted(self.document_types),
                'primary_publishing_organisation': sorted(self.primary_publishing_organisations),
                'publishing_app': sorted(self.publishing_apps)
            }, f)


class TextData:
    def __init__(self):
        self.titles = []
        self.descriptions = []
        self.combined_texts = []

    def tokenize_and_save(self):
        logger.info('tokenizing texts')
        create_and_save_tokenizer(
            self.combined_texts,
            num_words=20000,
            outfilename=OUTPUT_TEXT_TOKENIZER
        )

        logger.info('tokenizing title')
        create_and_save_tokenizer(
            self.titles,
            num_words=10000,
            outfilename=OUTPUT_TITLE_TOKENIZER
        )

        logger.info('tokenizing description')
        create_and_save_tokenizer(
            self.descriptions,
            num_words=10000,
            outfilename=OUTPUT_DESCRIPTION_TOKENIZER
        )


HEADER_LIST = [
    "base_path",
    "content_id",
    "document_type",
    "primary_publishing_organisation",
    "publishing_app",
    "title",
    "ordered_related_items",
    "quick_links",
    "related_mainstream_content",
    "related_guides",
    "document_collections",
    "slugs"
]


def process_content_item(content_item, clean_content_writer, content_to_taxon_map_writer, metadata, textdata):
    if content_item['locale'] != 'en':
        return

    # if content_item['document_type'] in (
    #     'worldwide_organisation',
    #     'placeholder_world_location_news_page',
    #     'travel_advice'
    # ):
    #     return # out-of-scope items

    content_item['title'] = extract_text(content_item["title"])
    content_item['description'] = extract_text(content_item["description"])
    content_item['body'] = get_text(content_item['details'])

    content_item['combined_text'] = " ".join(content_item[x] for x in ('title', 'description', 'body'))
    primary_publishing_organisation = get_primary_publishing_org(content_item)

    if primary_publishing_organisation:
        content_item['primary_publishing_organisation'] = primary_publishing_organisation['title']
        metadata.primary_publishing_organisations.add(primary_publishing_organisation['title'])

    try:
        content_item['ordered_related_items'] = [link['content_id'] for link in content_item['links']['ordered_related_items']]
    except KeyError:
        content_item['ordered_related_items'] = None

    try:
        content_item['quick_links'] = content_item['detail']['quick_links']
    except KeyError:
        content_item['quick_links'] = None

    try:
        content_item['related_mainstream_content'] = [link['content_id'] for link in content_item['links']['related_mainstream_content']]
    except KeyError:
        content_item['related_mainstream_content'] = None

    try:
        content_item['related_guides'] = [link['content_id'] for link in
                                          content_item['links']['related_guides']]
    except KeyError:
        content_item['related_guides'] = None

    try:
        content_item['document_collections'] = [link['content_id'] for link in
                                                content_item['links']['document_collections']]
    except KeyError:
        content_item['document_collections'] = None

    try:
        content_item['slugs'] = [part['slug'] for part in
                                 content_item['details']['parts']]
    except KeyError:
        content_item['slugs'] = None

    # Get content_id, taxon_id pairs and write to csv

    clean_content_writer.writerow([content_item.get(x) for x in HEADER_LIST])

    metadata.document_types.add(content_item['document_type'])
    metadata.publishing_apps.add(content_item['publishing_app'])

    textdata.titles.append(content_item['title'])


def clean_content():
    with open(OUTPUT_CLEAN_CONTENT, 'w') as f:
        clean_content_writer = csv.writer(f)
        clean_content_writer.writerow(HEADER_LIST)

        with open(OUTPUT_CONTENT_TO_TAXON_MAP, 'w') as f:
            content_to_taxon_map_writer = csv.writer(f)
            content_to_taxon_map_writer.writerow(["content_id", "taxon_id"])

            textdata = TextData()
            metadata = Metadata()
            progress_bar = jenkins_compatible_progress_bar()

            for content_item in progress_bar(items_from_content_file()):

                try:
                    process_content_item(
                        content_item,
                        clean_content_writer,
                        content_to_taxon_map_writer,
                        metadata,
                        textdata
                    )

                except Exception as e:
                    print(content_item)
                    print(e)
                    exit(1)

    metadata.write()
    textdata.tokenize_and_save()

    for filename in (
            OUTPUT_CLEAN_CONTENT,
            OUTPUT_CONTENT_TO_TAXON_MAP,
            OUTPUT_TEXT_TOKENIZER,
            OUTPUT_TITLE_TOKENIZER,
            OUTPUT_DESCRIPTION_TOKENIZER,
            OUTPUT_METADATA_LISTS
    ):
        os.rename(filename, os.path.splitext(filename)[0])


if __name__ == '__main__':
    clean_content()


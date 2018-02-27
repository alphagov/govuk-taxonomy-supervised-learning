"""Extract content from from data/content.json
"""
# coding: utf-8

import json
import logging.config
import os
import pathlib
from collections import OrderedDict

import pandas as pd
from lxml import html
from pandas.io.json import json_normalize

from pipeline_functions import extract_text, write_csv
from tokenizing import create_and_save_tokenizer

# Setup pipeline logging

LOGGING_CONFIG = os.getenv('LOGGING_CONFIG')
logging.config.fileConfig(LOGGING_CONFIG)
logger = logging.getLogger('clean_content')

# Get data file locations

DATADIR = os.getenv('DATADIR')
CONTENT_INPUT_FILE = 'raw_content.json.gz'
CONTENT_INPUT_PATH = os.path.join(DATADIR, CONTENT_INPUT_FILE)

DOCUMENT_TYPE_FILE = 'document_type_group_lookup.json'
DOCUMENT_TYPE_PATH = os.path.join(DATADIR, DOCUMENT_TYPE_FILE)

CONTENT_OUTPUT_FILE = 'clean_content.csv.gz'
CONTENT_OUTPUT_PATH = os.path.join(DATADIR, CONTENT_OUTPUT_FILE)

UNTAGGED_OUTPUT_FILE = 'untagged_content.csv.gz'
UNTAGGED_OUTPUT_PATH = os.path.join(DATADIR, UNTAGGED_OUTPUT_FILE)

# Assert that the file exists
try:
    assert os.path.exists(CONTENT_INPUT_PATH)
except AssertionError:
    logger.exception('%s does not exist', CONTENT_INPUT_PATH)
    raise

# Read in raw content file
logger.info('Importing data from %s.', CONTENT_INPUT_PATH)

# Convert to uri to satisfy pd.read_json

CONTENT_INPUT_PATH_URI = pathlib.Path(CONTENT_INPUT_PATH).as_uri()

content = pd.read_json(
    CONTENT_INPUT_PATH_URI,
    compression='gzip',
    orient='table',
    typ='frame',
    dtype=True,
    convert_axes=True,
    convert_dates=True,
    keep_default_dates=True,
    numpy=False,
    precise_float=False,
    date_unit=None
)

# Check content dataframe

try:
    assert content.shape[1] == 11
except AssertionError:
    logger.exception('Incorrect number of input columns')
    raise

try:
    assert content.shape[0] > 100000
except AssertionError:
    logger.warning('Less than 100,000 rows in raw content')

logger.info('Number of rows in raw content: %s.', content.shape[0])

logger.info('Number of duplicate content_ids in raw content: %s.',
            content[content.duplicated('content_id')].shape[0])
logger.debug('Printing head from content: %s.', content.head())

# Read in lookup table for document type group
logger.info('Importing document type group lookup from %s.',
            DOCUMENT_TYPE_PATH)

with open(DOCUMENT_TYPE_PATH, 'r') as fp:
    lookup_dict = json.load(fp)

docgp_lookup = pd.DataFrame.from_dict(lookup_dict, orient='index')
docgp_lookup.columns = ['document_type_gp']

# Merge lookup dataframe
content = pd.merge(
    left=content,
    right=docgp_lookup,
    left_on='document_type',
    right_index=True,
    how='outer',
    indicator=True
)

content.loc[content['_merge'] == 'left_only', 'document_type_gp'] = 'other'
content.drop('_merge', axis=1, inplace=True)


def is_json(raw_text):
    try:
        json_normalize(raw_text).columns.tolist()
    except AttributeError:
        return False
    return True


def is_html(raw_text):
    return html.fromstring(str(raw_text)).find('.//*') is not None


look = ['title', 'body']
child_keys = ['title','description']
filtered = ['body','brand','documents','final_outcome_detail','final_outcome_documents',
            'government','headers','introduction','introductory_paragraph',
            'licence_overview','licence_short_description','logo','metadata','more_information','need_to_know',
            'other_ways_to_apply','summary','ways_to_respond','what_you_need_to_know','will_continue_on','parts',
            'collection_groups']


def get_text(x):
    """
From dict to json and back (to OrderedDict), iterate over json from details column (based on list filtered, should
reconsider) and extract plaintext from included html.
    :param x: details cell from dataset
    :return: plaintext
    """
    total_text = ""
    string_json = json.dumps(OrderedDict(x))
    order_json = json.loads(string_json,object_pairs_hook=OrderedDict)
    for key,raw_text in sorted(order_json.items()):
        if key in filtered:
            if isinstance(raw_text,str) and len(raw_text)>1:
                raw_text = raw_text.replace("-"," ")
                raw_token = raw_text.split(" ")
                if len(raw_token)>0:
                    raw_string = extract_text(raw_text)
                    total_text += " " + raw_string
            elif isinstance(raw_text,list) and len(raw_text)>0:
                for sub_text in raw_text:
                    if is_json(sub_text):
                        total_text += nested_extract(sub_text)
                    elif is_html(sub_text):
                        str_from_html = extract_text(sub_text)
                        total_text += " " + str_from_html
    return total_text.strip()


def nested_extract(x):
    """
Iterate over nested json (avoiding recursion), flattening loops.
    :param x: nested `details` cell contents
    :return: plaintext
    """
    ttext = ""
    string_json2 = json.dumps(OrderedDict(x))
    order_json2 = json.loads(string_json2,object_pairs_hook=OrderedDict)
    if ('body' or 'title') in order_json2.keys():
        for item in look:
            raw_string2 = extract_text(order_json2[item])
            if len(raw_string2.split()) > 1:
                ttext += " " +raw_string2
    elif 'child_sections' in order_json2.keys():
        for child in order_json2['child_sections']:
            for key in child_keys:
                ttext += " " + child[key]
    return ttext

# Clean the html


logger.info('Extracting title, description, and text from content.')

logger.info('Extracting text from nested json tags (previously body)')
content['body'] = content['details'].map(get_text)
logger.debug('Text extracted from body looks like: %s', content['body'][0:10])

logger.info('Extracting text from description')
content = content.assign(description=content['description'].apply(extract_text))
logger.debug('Text extracted from description looks like: %s', content['description'][0:10])

logger.info('Extracting text from title')
content = content.assign(title=content['title'].apply(extract_text))
logger.debug('Text extracted from title looks like: %s', content['title'][0:10])

logger.info('Concatenating title, description, and text.')
content['combined_text'] = content['title'] + ' ' + content['description'] + ' ' + content['body']


# Filter out content not in english (locale =='en')

logger.info('Filtering out non-english documents')
logger.info('content.shape before filtering: %s', content.shape)

content = content[content.locale == 'en']

logger.info("content.shape after keeping only english content: %s", content.shape)


# stripout out-of-scope World items

logger.info('content shape before removing doctypes related to world %s', content.shape)
content = content[content.document_type != 'worldwide_organisation']
content = content[content.document_type != 'placeholder_world_location_news_page']
content = content[content.document_type != 'travel_advice']
logger.info('content shape after removing doctypes related to world %s', content.shape)


# TOKENIZE ALL content before splitting into labelled/unlabelled
logger.info('tokenizing combined_text')

create_and_save_tokenizer(
    content['combined_text'], 
    num_words=20000, 
    outfilename=os.path.join(DATADIR, "combined_text_tokenizer.json")
)

logger.info('tokenizing title')

create_and_save_tokenizer(
    content['title'], 
    num_words=10000, 
    outfilename=os.path.join(DATADIR, "title_tokenizer.json")
)

logger.info('tokenizing description')

create_and_save_tokenizer(
    content['description'], 
    num_words=10000, 
    outfilename=os.path.join(DATADIR, "description_tokenizer.json")
)



# Identify and select untagged content items

logger.info('Separating untagged content')

untagged = content[content['taxons'].isnull()]

logger.debug("Checking type of untagged['first_published_at']: %s",
             untagged['first_published_at'].dtype)

# Re-index untagged content items as date first published

logger.debug("Creating timeseries index on untagged['first_published_at']")

untagged = untagged.assign(first_published_at=pd.to_datetime(untagged['first_published_at']))

logger.debug("Checking type of untagged['first_published_at']: %s",
             untagged['first_published_at'].dtype)

logger.debug("Setting timestamp to index on untagged")
untagged.index = untagged['first_published_at']

# Save untagged content items

write_csv(untagged, 'Untagged content',
          UNTAGGED_OUTPUT_PATH, logger)

logger.info('Removing content with no taxons')

# TODO: Notebook version used content['taxons'] =
# content['taxons'].where((pd.notnull(content['taxons'])), None) needs testing
# to ensure that the alternative below works ok.

content = content[content['taxons'].notnull()]

logger.debug('content.columns: %s', content.columns)

# Save column names, excluding 'taxons' for later melt

content_columns = content.drop(['taxons'], axis=1).columns.values

logger.info('Creating content_wide dataframe')

# Concatenate columns (without taxons) with taxons that have been split
# into columns.

content_wide = pd.concat([content.drop('taxons', axis=1),
                          content['taxons'].apply(pd.Series)], axis=1)

logger.debug('content_wide[0:10]: %s', content_wide[0:10])
logger.debug('content_wide.shape: %s', content_wide.shape)
logger.info('Creating content_long dataframe')

content_long = pd.melt(content_wide, id_vars=content_columns, value_name='taxon')

logger.debug('content_long[0:10]: %s', content_long[0:10])
logger.debug('content_long.shape: %s', content_long.shape)
logger.debug('content_long.columns: %s', content_long.columns)

# Create mask to remove null taxons

logger.info('Create mask of null taxons')

mask = content_long['taxon'].isnull()

logger.debug('There are %s rows to be dropped', len(mask))

# Drop rows with null taxons

logger.info('Drop rows with null taxons.')

content_long = content_long[~mask]

logger.debug('content_long.shape: %s', content_long.shape)

logger.info('Extract content_id into taxon_id column.')

content_long = content_long.assign(taxon_id=[d['content_id'] for d in content_long['taxon']])

logger.debug("content_long['taxon'][0:10]: %s", content_long['taxon'][0:10])

content_long = content_long.drop(['taxon'], axis=1)

logger.debug('content_long.shape: %s', content_long.shape)
logger.debug('content_long.head(): %s', content_long.head())

# Assert content long has 14 columns

logger.info('Assert that column_long has the expected 15 columns.')

try:
    assert content_long.shape[1] == 15
except AssertionError:
    logger.exception('Incorrect number of columns in content_long (labelled content)')
    raise

# Assert content long has more than 300000 rows

logger.info('Assert that column_long has tmore than 300,000 rows.')

try:
    assert content_long.shape[0] > 300000
except AssertionError:
    logger.warning('Less than 300,000 rows in content_long (labelled content)')

# Confirm that untagged content is not contained in content_long

logger.info('Check that content_long has unique content_ids to untagged')

untagged_drop_check = pd.merge(
    left=content_long,
    right=untagged,
    on='content_id',
    how='outer',
    indicator=True
)

try:
    assert untagged_drop_check['taxon_id'][untagged_drop_check['_merge'] == 'both'].shape[0] < 10
except AssertionError:
    logger.exception('There are %s content items in both the untagged and labelled data',
                     untagged_drop_check['taxon_id'][untagged_drop_check['_merge'] == 'both'].shape[0])
    raise

# Write out to intermediate csv

write_csv(content_long, 'Long content dataframe',
          CONTENT_OUTPUT_PATH, logger)

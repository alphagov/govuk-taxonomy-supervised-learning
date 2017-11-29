"""Extract content from from data/content.json
"""
# coding: utf-8

import json
import io
import os
import pathlib
import logging
import logging.config
import numpy as np
import pandas as pd
from lxml import etree
from pipeline_functions import extract_text

# Setup pipeline logging

LOGGING_CONFIG = os.getenv('LOGGING_CONFIG')
logging.config.fileConfig(LOGGING_CONFIG)
logger = logging.getLogger('pipeline')

# Get data file locations

DATADIR = os.getenv('DATADIR')
CONTENT_INPUT_FILE = 'raw_content.json.gz'
CONTENT_INPUT_PATH = os.path.join(DATADIR, CONTENT_INPUT_FILE)

CONTENT_OUTPUT_FILE = 'clean_content.csv'
CONTENT_OUTPUT_PATH = os.path.join(DATADIR, CONTENT_OUTPUT_FILE)

UNTAGGED_OUTPUT_FILE = 'untagged_content.csv'
UNTAGGED_OUTPUT_PATH = os.path.join(DATADIR, UNTAGGED_OUTPUT_FILE)

# Convert to uri to satisfy pd.read_json


# Assert that the file exists

assert os.path.exists(CONTENT_INPUT_PATH), logger.error('%s does not exist', CONTENT_INPUT_PATH)

logger.info('Importing data from %s.', CONTENT_INPUT_PATH)

CONTENT_INPUT_PATH_URI = pathlib.Path(CONTENT_INPUT_PATH).as_uri()

content = pd.read_json(
    CONTENT_INPUT_PATH_URI, 
    compression = 'gzip',
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


logger.debug('Printing head from content: %s.', content.head())
logger.info('Extracting body from body dict')

content = content.assign(body = [d.get('body') for d in content.details])

logger.debug('Printing top 10 from content.body: %s.', content.body[0:10])

logger.info('Separating untagged content')

#Save untagged content items
untagged = content[content['taxons'].isnull()]

logger.debug("Checking type of untagged['first_published_at']: %s", untagged['first_published_at'].dtype)

logger.debug("Creating timeseries index on untagged['first_published_at']")

untagged = untagged.assign(first_published_at = pd.to_datetime(untagged['first_published_at']))

logger.debug("Checking type of untagged['first_published_at']: %s", untagged['first_published_at'].dtype)

logger.debug("Setting timestamp to index on untagged")
untagged.index = untagged['first_published_at'] 

logger.info('Saving untagged content to %s', UNTAGGED_OUTPUT_PATH)
untagged.to_csv(UNTAGGED_OUTPUT_PATH)
logger.debug('%s written to disk: %s', UNTAGGED_OUTPUT_PATH, os.path.exists('UNTAGGED_OUTPUT_PATH'))

# TODO confirm that untagged content later get dropped from content

# Clean the html

logger.info('Extracting text from body')
content = content.assign(body = content['body'].apply(extract_text))
logger.debug('Text extracted from body looks like: %s', content['body'][0:10])

logger.info('Extracting text from description')
content = content.assign(description = content['description'].apply(extract_text))
logger.debug('Text extracted from description looks like: %s', content['description'][0:10])

logger.info('Extracting text from title')
content = content.assign(title = content['title'].apply(extract_text))
logger.debug('Text extracted from title looks like: %s', content['title'][0:10])

logger.info('Concatenating title, description, and text.')
content['combined_text'] = content['title'] + ' ' + content['description'] + ' ' + content['body']

#logger.info('Dropping ')
content['taxons'] = content['taxons'].where((pd.notnull(content['taxons'])), None)

logger.debug('content.columns: %s',content.columns)

content_columns = content.drop(['taxons'], axis=1).columns.values

logger.info('Creating content_wide dataframe')
content_wide = pd.concat([content.drop('taxons', axis=1), content['taxons'].apply(pd.Series)], axis=1)
logger.debug('content_wide[0:10]: %s', content_wide[0:10])
logger.debug('content_wide.shape: %s', content_wide.shape)

logger.info('Creating content_long dataframe')
content_long = pd.melt(content_wide, id_vars=content_columns, value_name='taxon')

logger.debug('content_long[0:10]: %s', content_long[0:10])
logger.debug('content_long.shape: %s', content_long.shape)
logger.debug('content_long.columns: %s', 
             content_long.columns)

# Create mask to remove null taxons

logger.info('Create mask of null taxons')
mask = content_long['taxon'].isnull()
logger.debug('There are %s rows to be dropped', len(mask))

# Drop rows with null taxons

logger.info('Drop rows with null taxons.')
content_long = content_long[~mask]
logger.debug('content_long.shape: %s', content_long.shape)


logger.info('Extract content_id into taxon_id column.')
content_long = content_long.assign(taxon_id = [d['content_id'] for d in content_long['taxon']])
logger.debug("content_long['taxon'][0:10]: %s", content_long['taxon'][0:10])

logger.info('Extract content_id into taxon_id column.')
content_long = content_long.drop(['taxon'], axis=1)
logger.debug('content_long.shape: %s', content_long.shape)
logger.debug('content_long.head(): %s', content_long.head())

# Write out to intermediate csv

content_long.to_csv(CONTENT_OUTPUT_PATH)

logger.info('Content written to %s', CONTENT_OUTPUT_PATH)

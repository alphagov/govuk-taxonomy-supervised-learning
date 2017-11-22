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

# Setup pipeline logging

LOGGING_CONFIG = os.getenv('LOGGING_CONFIG')
logging.config.fileConfig(LOGGING_CONFIG)
logger = logging.getLogger('pipeline')

# Get data file locations

DATADIR = os.getenv('DATADIR')
CONTENT_INPUT = 'raw_content.json.gz'
CONTENT_OUTPUT = 'clean_content.csv'

# Convert to uri to satisfy pd.read_json

DATAPATH = os.path.join(DATADIR, CONTENT_INPUT)

# Assert that the file exists

assert os.path.exists(DATAPATH), logger.error('%s does not exist', DATADIR)

DATAPATH = pathlib.Path(DATAPATH).as_uri()

logger.info('Importing data from %s.', DATAPATH)

content = pd.read_json(
    DATAPATH, 
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

# Clean the html

def extract_text(body):
    """
    Extract text from html body

    :param body: <str> containing html.
    """
    #TODO: Tidy this up!
    r = None
    if body and body != '\n':
        tree = etree.HTML(body)
        r = tree.xpath('//text()')
        r = ' '.join(r)
        r = r.strip().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        r = r.replace('\n', ' ').replace(',', ' ')
        r = r.lower()
        r = ' '.join(r.split())
    if not r:
        r = ' '
    return r

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

OUTPATH = os.path.join(DATADIR, CONTENT_OUTPUT)
content_long.to_csv(OUTPATH)

logger.info('Content written to %s', OUTPATH)

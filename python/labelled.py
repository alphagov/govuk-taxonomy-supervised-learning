# coding: utf-8
'''Create labelled dataset from clean_content.csv and clean_taxons.csv
'''

import os
import logging
import logging.config
import pandas as pd

# Setup pipeline logging

LOGGING_CONFIG = os.getenv('LOGGING_CONFIG')
logging.config.fileConfig(LOGGING_CONFIG)
logger = logging.getLogger('pipeline')

# Get data file locations

DATADIR = os.getenv('DATADIR')
CONTENT_INPUT_FILE = 'clean_content.csv.gz'
CONTENT_INPUT_PATH = os.path.join(DATADIR, CONTENT_INPUT_FILE)

TAXONS_INPUT_FILE = 'clean_taxons.csv'
TAXONS_INPUT_PATH = os.path.join(DATADIR, TAXONS_INPUT_FILE)

UNTAGGED_OUTPUT_FILE = 'untagged_content.csv'
UNTAGGED_OUTPUT_PATH = os.path.join(DATADIR, UNTAGGED_OUTPUT_FILE)

EMPTY_TAXONS_OUTPUT_FILE = 'empty_taxons.csv'
EMPTY_TAXONS_OUTPUT_PATH = os.path.join(DATADIR, EMPTY_TAXONS_OUTPUT_FILE)

LABELLED_OUTPUT_FILE = 'labelled.csv'
LABELLED_OUTPUT_PATH = os.path.join(DATADIR, LABELLED_OUTPUT_FILE)

# Import clean_content (output py clean_content.py)

logger.info('Importing from %s as clean_content', CONTENT_INPUT_PATH)

clean_content = pd.read_csv(
    CONTENT_INPUT_PATH,
    compression='gzip'
    )

logger.info('clean_content.shape: %s.', clean_content.shape)
logger.debug('clean_content.head(): %s.', clean_content.head())

# Import clean_taxons (output by clean_taxons.csv)

logger.info('Importing from %s as clean_taxons', TAXONS_INPUT_PATH)

clean_taxons = pd.read_csv(
    TAXONS_INPUT_PATH
    )

logger.info('clean_taxons.shape: %s.', clean_taxons.shape)
logger.debug('clean_taxons.head(): %s.', clean_taxons.head())
logger.info('clean_taxons.columns: %s.', clean_taxons.columns)

logger.info('Dropping extraneous columns')

clean_taxons = clean_taxons[['base_path','content_id','taxon_name','level1taxon','level2taxon','level3taxon','level4taxon']].copy()

logger.info('clean_taxons.columns: %s.', clean_taxons.columns)
logger.info('clean_taxons.shape: %s.', clean_taxons.shape)


# Merge clean_content and clean_taxons to give labelled.

logger.info('Merging clean_content and clean_taxons into labelled')

labelled = pd.merge(
    left=clean_content,
    right=clean_taxons,
    left_on='taxon_id', # which taxon is the content item tagged to
    right_on='content_id', # what is the id of that taxon
    how='outer', # keep everything for checking merge
    indicator=True # so we can filter by match type
)

logger.info('labelled.shape: %s.', labelled.shape)
logger.debug('labelled.head(): %s.', labelled.head())

logger.info('Checking output of the merge: %s', labelled['_merge'].value_counts())
logger.info('labelled.columns: %s', labelled.columns)

labelled.rename(
    columns={'base_path_x': 'base_path', 'content_id_x': 'content_id'},
    inplace=True
)

logger.info('Checking unique content_ids from content without taxons (left_only) after merge: %s', labelled[labelled._merge == 'left_only'].content_id.nunique())

# Save out empty taxons (those which have no content tagged
# to them)

logger.info('Extracting empty taxons (right_only) after merge')

empty_taxons = labelled[labelled._merge == 'right_only']

logger.info('empty_taxons.shape: %s', empty_taxons.shape)
logger.info('Writing empty_taxons to %s', EMPTY_TAXONS_OUTPUT_PATH)

empty_taxons.to_csv(EMPTY_TAXONS_OUTPUT_PATH)

# Drop all data from labelled that did not join cleanly

logger.info('Retaining only perfect matches (_merge=both) in labelled')
logger.info('labelled.shape before dropping incomplete matches: %s', labelled.shape)

labelled = labelled[labelled._merge == 'both']

logger.info('labelled.shape after dropping incomplete matches: %s', labelled.shape)

logger.info('Dropping duplicate rows where content_id and taxon_id match')

labelled = labelled.drop_duplicates(subset=['content_id','taxon_id'])

logger.info('labelled.shape after dropping duplicates: %s', labelled.shape)
logger.info('Unique content_ids after dropping duplicates: %s', labelled.content_id.nunique())

logger.info('%s', labelled.columns)



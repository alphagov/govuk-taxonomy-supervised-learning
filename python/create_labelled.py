# coding: utf-8
'''Create labelled dataset from clean_content.csv and clean_taxons.csv
'''

import os
import logging.config
import pandas as pd
from pipeline_functions import write_csv

# Setup pipeline logging

LOGGING_CONFIG = os.getenv('LOGGING_CONFIG')
logging.config.fileConfig(LOGGING_CONFIG)
logger = logging.getLogger('create_labelled')

# Setup input file paths

DATADIR = os.getenv('DATADIR')
CONTENT_INPUT_PATH = os.path.join(DATADIR, 'clean_content.csv')
TAXONS_INPUT_PATH = os.path.join(DATADIR, 'clean_taxons.csv.gz')
CONTENT_TO_TAXON_MAP = os.path.join(DATADIR, 'content_to_taxon_map.csv')

# Set file output paths

LABELLED_OUTPUT_PATH = os.path.join(DATADIR, 'labelled.csv.gz')
UNTAGGED_OUTPUT_PATH = os.path.join(DATADIR, 'untagged.csv.gz')
EMPTY_TAXONS_OUTPUT_PATH = os.path.join(DATADIR, 'empty_taxons.csv.gz')
LABELLED_LEVEL1_OUTPUT_PATH = os.path.join(DATADIR, 'labelled_level1.csv.gz')
LABELLED_LEVEL2_OUTPUT_PATH = os.path.join(DATADIR, 'labelled_level2.csv.gz')

# Import clean_content (output by clean_content.py)

logger.info('Importing from %s as clean_content', CONTENT_INPUT_PATH)

clean_content = pd.read_csv(
    CONTENT_INPUT_PATH
    )

logger.info('clean_content.shape: %s.', clean_content.shape)
logger.debug('clean_content.head(): %s.', clean_content.head())

# Import content_to_taxon_map

logger.info('Importing from %s as clean_content', CONTENT_TO_TAXON_MAP)

content_to_taxon_map = pd.read_csv(CONTENT_TO_TAXON_MAP)

# Import clean_taxons (output by clean_taxons.csv)

logger.info('Importing from %s as clean_taxons', TAXONS_INPUT_PATH)

clean_taxons = pd.read_csv(
    TAXONS_INPUT_PATH,compression='gzip'
    )

logger.info('clean_taxons.shape: %s.', clean_taxons.shape)
logger.debug('clean_taxons.head(): %s.', clean_taxons.head())
logger.info('clean_taxons.columns: %s.', clean_taxons.columns)


# Merge clean_content and clean_taxons to create labelled data.

logger.info('Merging clean_content and clean_taxons into labelled')

content_taxons = pd.merge(
    left=clean_content,
    right=content_to_taxon_map,
    on='content_id',
    how='left'
)


labelled = pd.merge(
    left=content_taxons,
    right=clean_taxons,
    left_on='taxon_id', # which taxon is the content item tagged to
    right_on='content_id', # what is the id of that taxon
    how='outer', # keep everything for checking merge
    indicator=True # so we can filter by match type
)

# Print various check results to log

logger.info('labelled.shape: %s.', labelled.shape)
logger.debug('labelled.head(): %s.', labelled.head())

logger.debug('labelled.columns: %s', labelled.columns)
logger.debug('Checking output of the merge: %s', labelled['_merge'].value_counts())
logger.info('There are %s tagged content items/taxon combinations '
            'with a matching taxon', labelled['_merge'].value_counts()[2])
logger.info('There are %s tagged content items/taxon combinations '
            'without a matching taxon', labelled['_merge'].value_counts()[0])
logger.info('There are %s taxons with nothing tagged to them', labelled['_merge'].value_counts()[1])

# Rename columns after merge (some will have _x or _y appended if
# they are duplicated across merging dataframes).

labelled.rename(
    columns={'base_path_x': 'base_path', 'content_id_x': 'content_id' 
    , 'base_path_y': 'taxon_base_path'},
    inplace=True
)

logger.info('Checking unique content_ids from content without taxons '
            '(left_only) after merge: %s',
            labelled[labelled._merge == 'left_only'].content_id.nunique())

# Save out empty taxons (those which have no content tagged
# to them)

logger.info('Extracting empty taxons (right_only) after merge')

empty_taxons = labelled[labelled._merge == 'right_only']

logger.info('empty_taxons.shape: %s', empty_taxons.shape)

# Extract the data with no taxons (left_only) from above merge

# Untagged content
logger.info('Extracting untagged content: not classified in topic taxonomy')
untagged = labelled[
    ['base_path', 'content_id', 'document_type',
     'first_published_at', 'locale', 'primary_publishing_organisation',
     'publishing_app', 'title', 'description', 'combined_text', 'taxon_id', '_merge']]

untagged = untagged[untagged._merge == 'left_only']

untagged = untagged.drop(['_merge'], axis=1)

# Drop all data from labelled that did not join cleanly

logger.info('Retaining only perfect matches (_merge=both) in labelled')
logger.info('labelled.shape before dropping incomplete matches: %s', labelled.shape)

labelled = labelled[labelled._merge == 'both']

logger.info('labelled.shape after dropping incomplete matches: %s', labelled.shape)
logger.info('Dropping duplicate rows where content_id and taxon_id match')

# Drop duplicates

labelled = labelled.drop_duplicates(subset=['content_id','taxon_id'])

logger.info('labelled.shape after dropping duplicates: %s', labelled.shape)
logger.info('Unique content_ids after dropping duplicates: %s', labelled.content_id.nunique())


labelled = labelled.drop(['content_id_y','_merge'], axis=1)


# Create the labelled_level1 and labelled_level2 dfs

# Only keep rows where level1/level2 combination is unique
level2_dedup = labelled.drop_duplicates(subset = ['content_id', 'level1taxon', 'level2taxon']).copy()

# Replace erroneous date

level2_dedup['first_published_at'] = level2_dedup['first_published_at'].str.replace('0001-01-01', '2001-01-01')

logger.info('There were %s content item/taxons before removing duplicates',
            labelled.shape[0])

logger.info('There were %s content items, unique level2 taxon pairs after '
            'removing duplicates by content_id, level1taxon and level2taxon',
            level2_dedup.shape[0])

# Identify and drop rows where level2 is missing

mask = pd.notnull(level2_dedup['level2taxon'])
level1_tagged = level2_dedup[~mask].copy()

logger.info('There were %s content items only tagged to level1',
            level1_tagged.shape[0])

level2_tagged = level2_dedup[mask].copy()

logger.info('There are %s content items tagged to level2 or lower',
            level2_tagged.shape[0])

try:
    assert level1_tagged.shape[0] + level2_tagged.shape[0] == level2_dedup.shape[0]
except AssertionError:
    logger.exception('labelled_level1 + labelled_level2 does not equal total labelled')
    raise

# Write out dataframes

write_csv(level1_tagged, 'level1 tagged labelled',
          LABELLED_LEVEL1_OUTPUT_PATH, logger)

write_csv(level2_tagged, 'level2 tagged labelled',
          LABELLED_LEVEL2_OUTPUT_PATH, logger)

write_csv(labelled, 'labelled',
          LABELLED_OUTPUT_PATH, logger)

write_csv(untagged, 'untagged',
          UNTAGGED_OUTPUT_PATH, logger)

write_csv(empty_taxons, 'empty_taxons',
          EMPTY_TAXONS_OUTPUT_PATH, logger)

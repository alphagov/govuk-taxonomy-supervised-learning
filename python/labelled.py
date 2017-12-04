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

# Setup input file paths

DATADIR = os.getenv('DATADIR')
CONTENT_INPUT_PATH = os.path.join(DATADIR, 'clean_content.csv.gz')
TAXONS_INPUT_PATH = os.path.join(DATADIR, 'clean_taxons.csv')

# Set file output paths

UNTAGGED_OUTPUT_PATH = os.path.join(DATADIR, 'untagged_content.csv')
EMPTY_TAXONS_OUTPUT_PATH = os.path.join(DATADIR, 'empty_taxons.csv')
LABELLED_OUTPUT_PATH = os.path.join(DATADIR, 'labelled.csv')
FILTERED_OUTPUT_PATH = os.path.join(DATADIR, 'filtered.csv')
OLD_TAGS_OUTPUT_PATH = os.path.join(DATADIR, 'old_tags.csv')
EMPTY_TAXONS_OUTPUT_PATH = os.path.join(DATADIR, 'empty_tags.csv')
EMPTY_TAXONS_NOT_WORLD_OUTPUT_PATH = os.path.join(DATADIR, 'empty_tags_not_world.csv')

# Import clean_content (output by clean_content.py)

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

# Drop extraneous columns

logger.info('Dropping extraneous columns')

clean_taxons = clean_taxons[['base_path','content_id','taxon_name','level1taxon','level2taxon','level3taxon','level4taxon']].copy()

logger.info('clean_taxons.columns: %s.', clean_taxons.columns)
logger.info('clean_taxons.shape: %s.', clean_taxons.shape)

# Merge clean_content and clean_taxons to create labelled data.

logger.info('Merging clean_content and clean_taxons into labelled')

labelled = pd.merge(
    left=clean_content,
    right=clean_taxons,
    left_on='taxon_id', # which taxon is the content item tagged to
    right_on='content_id', # what is the id of that taxon
    how='outer', # keep everything for checking merge
    indicator=True # so we can filter by match type
)

# Print various check results to log

logger.info('labelled.shape: %s.', labelled.shape)
logger.debug('labelled.head(): %s.', labelled.head())

logger.info('labelled.columns: %s', labelled.columns)
logger.info('Checking output of the merge: %s', labelled['_merge'].value_counts())
logger.info('There are %s tagged content items/taxon combinations '
            'with a matching taxon', labelled['_merge'].value_counts()[2])
logger.info('There are %s tagged content items/taxon combinations '
            'without a matching taxon', labelled['_merge'].value_counts()[0])
logger.info('There are %s taxons with nothing tagged to them', labelled['_merge'].value_counts()[1])

# Rename columns after merge (some will have _x or _y appended if
# they are duplicated across merging dataframes).

labelled.rename(
    columns={'base_path_x': 'base_path', 'content_id_x': 'content_id'},
    inplace=True
)

logger.info('Checking unique content_ids from content without taxons '
            '(left_only) after merge: %s',
            labelled[labelled._merge == 'left_only'].content_id.nunique())

# Save out empty taxons (those which have no content tagged
# to them)

logger.info('Extracting empty taxons (right_only) after merge')

empty_taxons = labelled[labelled._merge == 'right_only']

logger.info('empty_taxons.shape: %s', empty_taxons.shape)
logger.info('Writing empty_taxons to %s', EMPTY_TAXONS_OUTPUT_PATH)

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

logger.info('%s', labelled.columns)

# Filter by taxon to exclude specific taxons from predictions

logger.info('Filtering by taxon to produced filtered_taxons')
logger.info('clean_taxons.shape before filtering: %s', clean_taxons.shape)

filtered_taxons = clean_taxons[clean_taxons.level1taxon != 'World']

logger.info("filtered_taxons.shape after filtering 'World' top taxons: %s", filtered_taxons.shape)

filtered_taxons = filtered_taxons[filtered_taxons.level1taxon != 'Corporate information']

logger.info("filtered_taxons.shape after filtering 'Corporate information' top taxons: %s", filtered_taxons.shape)

# Merge filtered taxons with content to create filtered

logger.info("Merging clean_content and filtered_taxons to create filtered")

filtered = pd.merge(
    left=clean_content,
    right=filtered_taxons,
    left_on='taxon_id',
    right_on='content_id',
    how='outer',
    indicator=True
)

logger.info("filtered.shape %s", filtered.shape)
logger.info("Checking output of the merge : %s", filtered['_merge'].value_counts())

logger.info("There are %s tagged content items/taxon combinations with "
            "a matching taxon", filtered['_merge'].value_counts()[2])
logger.info("There are %s tagged content items/taxon combinations "
            "without a matching taxon", filtered['_merge'].value_counts()[0])
logger.info("There are %s taxons with nothing tagged to them", filtered['_merge'].value_counts()[1])

empty_taxons_notworld = filtered[filtered._merge == 'right_only']

logger.info('empty_taxons_notworld.columns: %s', empty_taxons_notworld.columns)

# TODO investigate why the level5taxon column has been lost here.

empty_taxons_notworld = empty_taxons_notworld[
    ['base_path_y', 'content_id_y', 'taxon_name', 'level1taxon',
     'level2taxon', 'level3taxon', 'level4taxon']]

# Extract the data with no taxons (left_only) from above merge

# TODO: Check whether this should be filtered of labelled

content_old_taxons = filtered[
    ['base_path_x', 'content_id_x', 'document_type',
     'first_published_at', 'locale', 'primary_publishing_organisation',
     'publishing_app', 'title', 'taxon_id']]

content_old_taxons = content_old_taxons[filtered._merge == 'left_only']

logger.info("There are %s taxons represented in the %s content item/taxon "
            "combinations which have no corresponding taxon in the taxon data",
            content_old_taxons.taxon_id.nunique(), content_old_taxons.shape[0])

logger.info("There are %s content items/taxon combinations with missing taxon "
            "because these were removed during taxon_clean.py",
            content_old_taxons[content_old_taxons.taxon_id.isnull()].shape[0])

# Tidy the filtered dataframe

logger.info("Tidying the filtered dataframe")
logger.info("filtered.shape: %s", filtered.shape)
logger.info("filtered.columns: %s", filtered.columns)

filtered = filtered.drop(['Unnamed: 0', 'variable', 'base_path_y', 'content_id_y'], axis=1)

filtered.rename(columns={'base_path_x': 'base_path', 
                         'content_id_x': 'content_id'}, inplace=True)

logger.info("filtered.columns after tidying: %s", filtered.columns)

# Count duplicates

logger.info("There are %s rows in the data before filtering", filtered.shape[0])
logger.info("There are %s unique content items in the data before filtering", 
            filtered.content_id.nunique())

# Drop any rows that were not perfectly matched in filtered taxons
# and content. But first record shape/duplicates for later comparison

filtered_rows = filtered.shape[0]
filtered_unique = filtered.content_id.nunique()
filtered_dupes = filtered[filtered.duplicated(['content_id', 'taxon_id'])].shape[0]

filtered = filtered[filtered._merge == 'both']

logger.info("There are %s rows in the taxon-level data after filtering out mismatches",
            filtered.shape[0])
logger.info("There are %s unique content items in the taxon-level data after filtering "
            "out mismatches", filtered.content_id.nunique())
logger.info("There were %s rows dropped because of mismatching",
            filtered_rows - filtered.shape[0])
logger.info("There were %s unique content items dropped because of mismatching",
            filtered_unique - filtered.content_id.nunique())

logger.info("Before removing mismatches, there were %s duplicates content items, "
            "both with matching content_id and taxon_id",
            filtered_dupes)

logger.info("After removing mismatches, there were %s duplicates content items, "
            "both with matching content_id and taxon_id",
            filtered[filtered.duplicated(['content_id', 'taxon_id'])].
            shape[0])

# Drop duplicates
logger.info("Dropping duplicates from filtered")
logger.info("filtered.shape before deduplication: %s", filtered.shape)

pre_dedup_rows = filtered.shape[0]
pre_dedup_unique = filtered.content_id.nunique()

filtered = filtered.drop_duplicates(subset = ['content_id', 'taxon_id'])

logger.info("There were %s additional rows dropped due to duplicate "
            "content_id/taxon_id combination",
            pre_dedup_rows - filtered.shape[0])

logger.info("There were %s additional content items dropped due to duplicate "
            "content_id/taxon_id combination",
            pre_dedup_unique - filtered.content_id.nunique())

logger.info("filtered.shape after deduplication: %s", filtered.shape)

# Write out dataframes


if os.path.exists(LABELLED_OUTPUT_PATH):
    logger.warning('Overwriting %s', LABELLED_OUTPUT_PATH)

logger.info("Saving labelled to %s", LABELLED_OUTPUT_PATH)
labelled.to_csv(LABELLED_OUTPUT_PATH)


if os.path.exists(FILTERED_OUTPUT_PATH):
    logger.warning('Overwriting %s', FILTERED_OUTPUT_PATH)

logger.info("Saving filtered to %s", FILTERED_OUTPUT_PATH)
filtered.to_csv(FILTERED_OUTPUT_PATH)


if os.path.exists(OLD_TAGS_OUTPUT_PATH):
    logger.warning('Overwriting %s', OLD_TAGS_OUTPUT_PATH)

logger.info("Saving old_taxons to %s", OLD_TAGS_OUTPUT_PATH)
content_old_taxons.to_csv(OLD_TAGS_OUTPUT_PATH)

# NOTE: I have saved this dataframe out here. In previous versions
# it was getting overwritten by the empty taxons csv.


if os.path.exists(EMPTY_TAXONS_NOT_WORLD_OUTPUT_PATH):
    logger.warning('Overwriting %s', EMPTY_TAXONS_NOT_WORLD_OUTPUT_PATH)

logger.info("Saving empty_taxons_not_world to %s", EMPTY_TAXONS_NOT_WORLD_OUTPUT_PATH)
empty_taxons_notworld.to_csv(EMPTY_TAXONS_NOT_WORLD_OUTPUT_PATH)

if os.path.exists(EMPTY_TAXONS_OUTPUT_PATH):
    logger.warning('Overwriting %s', EMPTY_TAXONS_OUTPUT_PATH)

logger.info("Saving empty_taxons data to %s", EMPTY_TAXONS_OUTPUT_PATH)
empty_taxons_notworld.to_csv(EMPTY_TAXONS_OUTPUT_PATH)

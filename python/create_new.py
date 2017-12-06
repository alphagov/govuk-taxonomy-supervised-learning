# coding: utf-8
'''Create labelled dataset from clean_content.csv and clean_taxons.csv
'''

import os
import logging
import logging.config
import pandas as pd
from pipeline_functions import write_csv

# Setup pipeline logging

LOGGING_CONFIG = os.getenv('LOGGING_CONFIG')
logging.config.fileConfig(LOGGING_CONFIG)
logger = logging.getLogger('create_labelled')

# Setup input file paths

DATADIR = os.getenv('DATADIR')
UNTAGGED_INPUT_PATH = os.path.join(DATADIR, 'untagged_content.csv')
OLD_TAXONS_INPUT_PATH = os.path.join(DATADIR, 'old_taxons.csv')

# Set file output paths

NEW_OUTPUT_PATH = os.path.join(DATADIR, 'new_content.csv')


# Import clean_content (output by clean_content.py)

logger.info('Importing from %s as untagged', UNTAGGED_INPUT_PATH)

untagged = pd.read_csv(
    UNTAGGED_INPUT_PATH
    )

logger.info('untagged.shape: %s.', untagged.shape)
logger.debug('untagged.head(): %s.', untagged.head())

# Import old_taxons, content which has been tagged to taxons not represented in the V1 topic taxonomy

logger.info('Importing from %s as old_taxons', OLD_TAXONS_INPUT_PATH)

old_taxons = pd.read_csv(
    OLD_TAXONS_INPUT_PATH
    )

logger.info('old_taxons.shape: %s.', old_taxons.shape)
logger.debug('old_taxons.head(): %s.', old_taxons.head())
logger.info('old_taxons.columns: %s.', old_taxons.columns)


# TODO: Add this to untagged data with a flag
logger.info("There are %s untagged rows", untagged.shape[0])
logger.info("There are %s untagged content items", 
            untagged.content_id.nunique())
logger.info("There are %s old_taxons rows", old_taxons.shape[0])
logger.info("There are %s old_taxons content items", 
            old_taxons.content_id.nunique())



# Write out dataframes

write_csv(new, 'new content',
          NEW_OUTPUT_PATH, logger)


"""Extract content from from data/content.json
"""
# coding: utf-8

import json
import io
import os
import logging
import logging.config
import numpy as np
import pandas as pd
import pathlib

# Setup pipeline logging

LOGGING_CONFIG = os.getenv('LOGGING_CONFIG')
logging.config.fileConfig(LOGGING_CONFIG)
logger = logging.getLogger('pipeline')

# Get data file locations

DATADIR = os.getenv('DATADIR')
CONTENT_INPUT = 'raw_content.json'
CONTENT_OUTPUT = 'clean_content.csv'

# Convert to uri to satisfy pd.read_json

DATAPATH = os.path.join(DATADIR, CONTENT_INPUT)

# Assert that the file exists

assert os.path.exists(DATAPATH), logger.error('%s does not exist', DATADIR)

DATAPATH = pathlib.Path(DATAPATH).as_uri()

logger.info('Importing data from %s.', DATAPATH)

#download the taxon data from content store for all links which are taxons
content = pd.read_json(
    DATAPATH, 
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

logger.info('content shape is %s: ', content.shape)
logger.debug('Printing content head %s: ', content.shape)
logger.debug('Printing content columns %s: ', content.columns)
logger.debug('Details column type is %s: ', type(content['details'][0]))
logger.debug("content['details'] looks like %s", content['details'][0])

# Write out to intermediate csv

OUTPATH = os.path.join(DATADIR, CONTENT_OUTPUT)
content.to_csv(OUTPATH)

logger.info('Content written to %s', OUTPATH)

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
FILENAME = 'content.json'

# Convert to uri to satisfy pd.read_json

DATAPATH = os.path.join(DATADIR, FILENAME)

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
logger.info('Printing content head %s: ', content.shape)


content.head()


# In[252]:


list(content.columns.values)


# In[254]:


type(content['details'][0])


# In[255]:


content['details'][0]


# In[256]:


content['details'][100]


# In[257]:


content['details'][10000]


# ## Plan
# - Sort out text variable, will be a concatenation of title, description and details->body
# - Get long by taxon
# - many-to-one merge
# - clean up column names
# 
# - check missings
# - check dupes
# 
# - Do some QA of merges and reshapes. e.g. shapes before/after
# 
# - Export in format for EDA
# 

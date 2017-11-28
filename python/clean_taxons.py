# coding: utf-8
'''
Extract taxon hierarchy from data/taxons.json
'''

import os
import pathlib
import logging
import logging.config
import numpy as np
import pandas as pd
from pipeline_functions import (ancestors, pushna, 
    conjunction)

# Setup pipeline logging

LOGGING_CONFIG = os.getenv('LOGGING_CONFIG')
logging.config.fileConfig(LOGGING_CONFIG)
logger = logging.getLogger('pipeline')

# Get data file locations

DATADIR = os.getenv('DATADIR')
TAXON_INPUT = 'raw_taxons.json'
TAXON_OUTPUT = 'clean_taxons.csv'
DATAPATH = os.path.join(DATADIR, TAXON_INPUT)

# Assert that the file exists

assert os.path.exists(DATAPATH), logger.error('%s does not exist', DATADIR)

# Convert to uri to satisfy pd.read_json

DATAPATH = pathlib.Path(DATAPATH).as_uri()

logger.info('Importing taxons from %s as taxons', DATAPATH)

# Load taxons

taxons = pd.read_json(
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

logger.info('taxon.shape: %s', taxons.shape[0])
logger.debug('Printing top 5 rows of taxons: %s', taxons.head())

# Convert null taxons to None and zip together into a dict

taxons_notnan = taxons.where(cond=(pd.notnull(taxons)), other=None)

logger.info('Creating child_dict')

child_dict = dict(zip(taxons_notnan['content_id'], 
                      taxons_notnan['parent_content_id']))

logger.debug('Printing top 5 keys from child_dict: %s',
             [i for i in child_dict.keys()][:5])

# Create dictionary containg taxon id (content_id) and the taxon path
# within taxonomy

logger.info('Iterating through child_dict to recursively generate child -> taxon path')

dict_taxonpath = {
    parent_content_id: ancestors(parent_content_id, child_dict)
    for parent_content_id in child_dict.keys()
}

logger.debug('Printing top 5 keys from dict_taxonpath: %s',
             [i for i in dict_taxonpath.keys()][:5])

# Convert dict to pandas dataframe

logger.info('Convert child_dict to pandas dataframe: taxonpath.')

# Note that in python3.x the dict needs to be converted to a list.
# Otherwise use pd.Dataframe.from_dict.

taxonpath = pd.DataFrame(
    list(dict_taxonpath.items()),
    columns=['content_id', 'taxonpath']
)

logger.debug('taxonpath.shape: %s', taxonpath.shape)
logger.info('The longest taxonpath is %s.',
            max(taxonpath['taxonpath'].map(len)))

# Get this taxonpath list into separate columns per taxon,
# within the taxon path, reverse the order of the list so
# Left = higher up tree.

logger.info('Separating taxon path into one column per taxon: split_taxonpath_to_cols')

split_taxonpath_to_cols = pd.concat(
     [taxonpath['content_id'],
     taxonpath['taxonpath'].apply(pd.Series).loc[:, ::-1]],
    axis=1
    )

logger.debug('split_taxonpath_to_cols.shape: %s', split_taxonpath_to_cols)

# TODO: Remove hard coding of split_taxonpath_to_cols below to generalise over deeper taxonomies

split_taxonpath_to_cols.columns = ['content_id', 'level1', 'level2',
         'level3', 'level4']

# Move non empty cells to left in grouped columns pandas:
# https://stackoverflow.com/questions/39361839/move-non-empty-cells-to-left-in-grouped-columns-pandas/39362818#39362818

logger.info('Moving empty cells to left in split_taxonpath_to_cols')

coltype = split_taxonpath_to_cols
coltype = coltype.columns
coltype = coltype.to_series().str
coltype = coltype.extract(r'(\D*)', expand=False)

logger.debug('Extracting coltype from split_taxonpath_to_cols: %s', coltype)

split_taxonpath_ordered = split_taxonpath_to_cols.groupby(coltype, axis=1)
split_taxonpath_ordered = split_taxonpath_ordered.apply(lambda split_taxonpath_to_cols: split_taxonpath_to_cols.apply(pushna, axis=1))

# lookup dictionary to get titles from the content_ids.
# Although content_ids remain gold standard data quality for manipulation,
# titles will be used during descriptive analysis so both will be retained
# for now.

logger.info('Creating id_tit_dict as a lookup between content id and titles.')

id_tit_dict = dict(zip(taxons_notnan['content_id'], taxons_notnan['title']))

logger.debug('Printing top 5 keys from id_tit_dict: %s',
             [i for i in id_tit_dict.keys()][:5])
# Pull titles into the content_id df, using the lookup dict
# TODO: Remove hardcoding to allow this to generalise over many columns of
# an arbitrarily deep taxonomy.
logger.info('Creating split_taxonpath_ordered')

split_taxonpath_ordered['contenttitle'] = split_taxonpath_ordered['content_id'].map(id_tit_dict)
split_taxonpath_ordered['level1taxon'] = split_taxonpath_ordered['level1'].map(id_tit_dict)
split_taxonpath_ordered['level2taxon'] = split_taxonpath_ordered['level2'].map(id_tit_dict)
split_taxonpath_ordered['level3taxon'] = split_taxonpath_ordered['level3'].map(id_tit_dict)
split_taxonpath_ordered['level4taxon'] = split_taxonpath_ordered['level4'].map(id_tit_dict)

# Merge to get the base path

logger.info('Merging taxons with split_taxonpath_ordered to create df_taxons')

df_taxons = pd.merge(
    taxons,
    split_taxonpath_ordered,
    how='left',
    on='content_id',
    indicator=True
)

logger.debug('df_taxons.shape: %s', df_taxons.shape)
logger.debug('Printing df_taxons.columns before drop: %s', list(df_taxons.columns.values))

df_taxons.drop(['parent_content_id', 'contenttitle', '_merge'], axis=1, inplace=True)
df_taxons.rename(columns={'title': 'taxon_name', 'level1_y': 'level1tax_id', 
    'level2': 'level2tax_id', 'level3': 'level3tax_id',
    'level4': 'level4tax_id'}, 
    inplace=True)

# For top taxons (level1) ensure that taxon)name is in level1taxon column instead of Nan
df_taxons['level1taxon'] = df_taxons['level1taxon'].fillna(df_taxons['taxon_name'])

taxonslevels = df_taxons.copy()

# Define the condition 

cond = conjunction(
    taxonslevels['level2taxon'].isna(), 
    taxonslevels['level1taxon'] != taxonslevels['taxon_name']
    )

# Change the values of the column if the condition is met to the 
# taxon-name, otherwise the original string

taxonslevels['level2taxon'] = np.where(cond, taxonslevels['taxon_name'], 
                                       taxonslevels['level2taxon'])

cond = conjunction(
    df_taxons['level2taxon'] != df_taxons['taxon_name'],
    df_taxons['level3taxon'].isna(),
    df_taxons['level2taxon'].notnull()
    )

taxonslevels['level3taxon'] = np.where(cond, taxonslevels['taxon_name'],
                                       taxonslevels['level3taxon'])

cond = conjunction(
    df_taxons['level3taxon'] != df_taxons['taxon_name'],
    df_taxons['level2taxon'] != df_taxons['taxon_name'],
    df_taxons['level4taxon'].isna(),
    df_taxons['level3taxon'].notnull()
    )

taxonslevels['level4taxon'] = np.where(
    cond, taxonslevels['taxon_name'], taxonslevels['level4taxon']
    )

# create new column for last taxon level
taxonslevels['level5taxon'] = np.nan
cond = conjunction(
    df_taxons['level4taxon'] != df_taxons['taxon_name'],
    df_taxons['level3taxon'] != df_taxons['taxon_name'],
    df_taxons['level2taxon'] != df_taxons['taxon_name'],
    df_taxons['level4taxon'].notnull()
    )

taxonslevels['level5taxon'] = np.where(cond, taxonslevels['taxon_name'], 
                                       taxonslevels['level5taxon'])

#copy the working df back to taxons
df_taxons = taxonslevels.copy()

logger.debug('Print df_taxons.columns after drop: %s', list(df_taxons.columns.values))

OUTPATH = os.path.join(DATADIR, TAXON_OUTPUT)
df_taxons.to_csv(OUTPATH)
logger.info('Taxons written to %s', OUTPATH)

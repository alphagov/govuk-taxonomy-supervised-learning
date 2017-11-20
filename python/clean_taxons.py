"""Extract taxon hierarchy from data/taxons.json
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
TAXON_INPUT = 'raw_taxons.json'
TAXON_OUTPUT = 'clean_taxons.csv'

# Convert to uri to satisfy pd.read_json

DATAPATH = os.path.join(DATADIR, TAXON_INPUT)

# Assert that the file exists

assert os.path.exists(DATAPATH), logger.error('%s does not exist', DATADIR)

DATAPATH = pathlib.Path(DATAPATH).as_uri()

logger.info('Importing data from %s.', DATAPATH)

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

logger.info('%s taxons loaded', taxons.shape[0])
logger.debug('Printing top 5 rows of taxons: %s', taxons.head())

taxons_notnan = taxons.where((pd.notnull(taxons)), None)
child_dict = dict(zip(taxons_notnan['content_id'], taxons_notnan['parent_content_id']))

# Define a function that recursively generates the child -> tree taxon path

def ancestors(parent_content_id):
    """
    Recursively generate child -> taxon path
    """
    parent = child_dict[parent_content_id]

    if parent is None:
        out = []
    else:
        out = [parent] + ancestors(parent)
    return out

# Create dictionary containg taxon id (content_id) and the taxon path 
# within taxonomy

logger.info('Iterating through child_dict')

dict_taxonpath = {
    parent_content_id: ancestors(parent_content_id)
    for parent_content_id in child_dict.keys()
}

logger.debug('Printing top 5 keys from dict_taxonpath: %s', 
        [i for i in dict_taxonpath.keys()][:5])

# Convert dict to pandas dataframe

logger.info('Convert child_dict to pandas dataframe.')

# Note that in python3.x the dict needs to be converted to a list. 
# Otherwise use pd.Dataframe.from_dict.

taxonpath = pd.DataFrame(
    list(dict_taxonpath.items()),
    columns=['content_id', 'taxonpath']
)

logger.info('The longest taxonpath is %s.', 
            max(taxonpath['taxonpath'].map(len)))


# Get this taxonpath list into separate columns per taxon, 
# within the taxon path, reverse the order of the list so 
# Left = higher up tree.


logger.info('Separating taxon path into one column per taxon')


split_taxonpath_to_cols = pd.concat([taxonpath['content_id'], 
     taxonpath['taxonpath'].apply(pd.Series).loc[:, ::-1]], axis=1)

# Hard coded so think about what to do if taxonomy gets deeper

split_taxonpath_to_cols.columns = ['content_id', 'level1', 'level2',
         'level3', 'level4'] 

# Move non empty cells to left in grouped columns pandas: 
# https://stackoverflow.com/questions/39361839/move-non-empty-cells-to-left-in-grouped-columns-pandas/39362818#39362818

# pushna Pushes all null values to the end of the series
# coltype Uses regex to extract the non-numeric prefix from all column names

def pushna(series):
    """
    Push all null values to the end of the series
    """
    notnull = series[series.notnull()]
    isnull = series[series.isnull()]
    values = notnull.append(isnull).values
    return pd.Series(values, series.index)

logger.info('Extracting coltype')

coltype = split_taxonpath_to_cols.columns.to_series().str.extract(r'(\D*)', expand=False)

logger.debug('coltype: %s', coltype)

split_taxonpath_ordered = split_taxonpath_to_cols.groupby(coltype, axis=1)
split_taxonpath_ordered = split_taxonpath_ordered.apply(lambda split_taxonpath_to_cols: 
        split_taxonpath_to_cols.apply(pushna, axis=1))

# lookup dictionary to get titles from the content_ids.
# Although content_ids remain gold standard data quality for manipulation, 
# titles will be used during descriptivve analysis so both will be retained 
# for now.

id_tit_dict = dict(zip(taxons_notnan['content_id'], taxons_notnan['title']))

#Pull titles into the content_id df, using the lookup dict
#hard coded so think about what to do if taxonomy gets deeper

split_taxonpath_ordered['contenttitle'] = split_taxonpath_ordered['content_id'].map(id_tit_dict)
split_taxonpath_ordered['level1taxon'] = split_taxonpath_ordered['level1'].map(id_tit_dict)
split_taxonpath_ordered['level2taxon'] = split_taxonpath_ordered['level2'].map(id_tit_dict)
split_taxonpath_ordered['level3taxon'] = split_taxonpath_ordered['level3'].map(id_tit_dict)
split_taxonpath_ordered['level4taxon'] = split_taxonpath_ordered['level4'].map(id_tit_dict)

# use merge to get the base path

df_taxons = pd.merge(
    taxons,
    split_taxonpath_ordered,
    how='left',
    on='content_id',
    indicator=True
)


logger.debug('Print df_taxons.columns before drop: %s', list(df_taxons.columns.values))



df_taxons.drop(['parent_content_id', 'contenttitle', '_merge'], axis=1, inplace=True)
df_taxons.rename(columns={'title': 'taxon_name', 'level1_y': 'level1tax_id', 'level2': 'level2tax_id',
    'level3': 'level3tax_id', 'level4': 'level4tax_id'}, inplace=True)


logger.debug('Print df_taxons.columns after drop: %s', list(df_taxons.columns.values))

OUTPATH = os.path.join(DATADIR, TAXON_OUTPUT)
df_taxons.to_csv(OUTPATH)

logger.info('Taxons written to %s', OUTPATH)

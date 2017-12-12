'''
Helper functions used in the clean_content.py and clean_taxons.py scripts.
'''
import os.path
import functools
import pandas as pd
import numpy as np
from lxml import etree

def write_csv(dataframe, name, path, logger, index=False, **kwargs):
    '''
    Write a dataframe to CSV with logging

    :param dataframe: <str> A pandas dataframe to be written out.
    Note that this is passed as a sring and then evaluated as an
    object to complete the logging message
    :param name: <str> Name of the object being written out.
    This is a description that is is written to the logging message
    :param path: <str> The path to be written to
    :param logger: <logging.getLogger> Current logger
    '''

    if os.path.exists(path):
        logger.warning('Overwriting %s', path)

    logger.info('Writing %s to %s', name, path)

    try:

        dataframe.to_csv(path, index=index, **kwargs)

    except Exception:
        logger.exception('Error writing %s to %s', name, path)
        raise


# TODO: Remove recursive function reference!
def ancestors(parent_content_id, child_dict):
    '''
    Recursively generate child -> taxon path
    '''
    parent = child_dict[parent_content_id]

    if parent is None:
        out = []
    else:
        out = [parent] + ancestors(parent, child_dict)
    return out

def pushna(series):
    '''
    Push all null values to the end of the series
    '''
    notnull = series[series.notnull()]
    isnull = series[series.isnull()]
    values = notnull.append(isnull).values
    out = pd.Series(values, series.index)
    return out

def conjunction(*args):
    '''
    Combine boolean series into one

    Take an arbitrary number or conditions, and combine the truth
    conditions. This is the equivalent of running:
    (x > 10) & (x < 20).

    :param args: <pd.Series> Logical condition applied to a pandas
    series or numpy array.
    '''

    mask = functools.reduce(np.logical_and, args)
    return mask

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

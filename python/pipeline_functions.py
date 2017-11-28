'''
Helper functions used in the clean_content.py and clean_taxons.py scripts.
'''
import pandas as pd
import numpy as np
import functools
from lxml import etree

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

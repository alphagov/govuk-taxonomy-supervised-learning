'''
Helper functions used in the clean_content.py and clean_taxons.py scripts.
'''
import os.path
import functools
import pandas as pd
import numpy as np
from lxml import etree
import json
from collections import OrderedDict
from pandas.io.json import json_normalize
from lxml import html



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

        dataframe.to_csv(path, index=index, compression='gzip', **kwargs)

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
    if body and body!="\n":
        try:
            tree = etree.HTML(body)
            r = tree.xpath('//text()')
            r = ' '.join(r)
            r = r.strip().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            r = r.replace('\n', ' ').replace(',', ' ')
            r = r.lower()
            r = ' '.join(r.split())
        except ValueError:
            print("exception @ extract:",type(body),body)
    if not r:
        r = ' '
    return r

def map_content_id_to_taxon_id(content_item):
    return [
        (content_item['content_id'], taxon['content_id'])
        for taxon in content_item['links'].get('taxons', [])
    ]



def get_primary_publishing_org(content_item):

    if 'primary_publishing_organisation' in content_item['links']:
        return content_item['links']['primary_publishing_organisation'][0]
    else:
        return None

look = ['title', 'body']
child_keys = ['title', 'description']
filtered = ['body', 'brand', 'documents', 'final_outcome_detail', 'final_outcome_documents',
            'government', 'headers', 'introduction', 'introductory_paragraph',
            'licence_overview', 'licence_short_description', 'logo', 'metadata', 'more_information', 'need_to_know',
            'other_ways_to_apply', 'summary', 'ways_to_respond', 'what_you_need_to_know', 'will_continue_on', 'parts',
            'collection_groups']

def get_text(x):
    """
From dict to json and back (to OrderedDict), iterate over json from details column (based on list filtered, should
reconsider) and extract plaintext from included html.
    :param x: details cell from dataset
    :return: plaintext
    """
    total_text = ""
    string_json = json.dumps(OrderedDict(x))
    order_json = json.loads(string_json, object_pairs_hook=OrderedDict)
    for key, raw_text in sorted(order_json.items()):
        if key in filtered:
            if isinstance(raw_text, str) and len(raw_text) > 1:
                raw_text = raw_text.replace("-", " ")
                raw_token = raw_text.split(" ")
                if len(raw_token) > 0:
                    raw_string = extract_text(raw_text)
                    total_text += " " + raw_string
            elif isinstance(raw_text, list) and len(raw_text) > 0:
                for sub_text in raw_text:
                    if is_json(sub_text):
                        total_text += nested_extract(sub_text)
                    elif is_html(sub_text):
                        str_from_html = extract_text(sub_text)
                        total_text += " " + str_from_html
    return total_text.strip()


def nested_extract(x):
    """
Iterate over nested json (avoiding recursion), flattening loops.
    :param x: nested `details` cell contents
    :return: plaintext
    """
    ttext = ""
    string_json2 = json.dumps(OrderedDict(x))
    order_json2 = json.loads(string_json2, object_pairs_hook=OrderedDict)
    if ('body' or 'title') in order_json2.keys():
        for item in look:
            raw_string2 = extract_text(order_json2[item])
            if len(raw_string2.split()) > 1:
                ttext += " " + raw_string2
    elif 'child_sections' in order_json2.keys():
        for child in order_json2['child_sections']:
            for key in child_keys:
                ttext += " " + child[key]
    return ttext

def is_json(raw_text):
    try:
        json_normalize(raw_text).columns.tolist()
    except AttributeError:
        return False
    return True


def is_html(raw_text):
    return html.fromstring(str(raw_text)).find('.//*') is not None

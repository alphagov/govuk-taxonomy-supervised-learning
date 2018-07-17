import os
import yaml
import gzip
import ijson

from data.json import stream_json

document_types_excluded_from_the_topic_taxonomy_filename = \
    os.path.join(os.path.abspath(os.path.dirname(__file__)),
                 '..', 'config', 'document_types_excluded_from_the_topic_taxonomy.yml'
    )

def document_types_excluded_from_the_topic_taxonomy():
    with open(
        document_types_excluded_from_the_topic_taxonomy_filename,
        'r'
    ) as f:
        return yaml.load(f)['document_types']

def items_from_content_file(datadir=None, filename="content.json.gz"):
    if datadir is None:
        datadir = os.getenv("DATADIR") or "data"

    full_filename = os.path.join(datadir, filename)

    with gzip.open(full_filename, mode='rt') as content_file:
        yield from ijson.items(content_file, prefix='item')

def write_content_file(filename, content, datadir=None):
    if datadir is None:
        datadir = os.getenv("DATADIR") or "data"

    full_filename = os.path.join(datadir, filename)

    with gzip.open(full_filename, 'wt') as output_file:
        stream_json(output_file, content)


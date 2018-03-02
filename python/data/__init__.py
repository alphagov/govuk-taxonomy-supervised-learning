import os
import yaml

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

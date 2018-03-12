# coding: utf-8

import logging.config
import os
import warnings

import pandas as pd
from sklearn.exceptions import DataConversionWarning

import dataprep
import tokenizing

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

DATADIR = os.getenv('DATADIR')

if __name__ == "__main__":

    LOGGING_CONFIG = os.getenv('LOGGING_CONFIG')
    logging.config.fileConfig(LOGGING_CONFIG)
    logger = logging.getLogger('create_new')

    logger.info("Loading data")
    new_content = pd.read_csv(
        os.path.join(DATADIR, 'new_content.csv.gz'),
        dtype=object,
        compression='gzip'
    )

    logger.info("Dropping columns")

    new_content.drop(['content_purpose_document_supertype', 'content_purpose_subgroup',
                      'content_purpose_supergroup', 'email_document_supertype',
                      'government_document_supertype', 'navigation_document_supertype',
                      'public_updated_at', 'search_user_need_document_supertype',
                      'taxon_id', 'user_journey_document_supertype', 'updated_at'], axis=1, inplace=True)

    # **** VECTORIZE META ********************
    # ************************************

    logger.info("Vectorizing metadata")
    meta = dataprep.create_meta(new_content['content_id'], new_content)

    # **** TOKENIZE TEXT ********************
    # ************************************

    # Load tokenizers, fitted on both labelled and unlabelled data from file
    # created in clean_content.py

    logger.info("Tokenizing combined_text")

    combined_text_sequences_padded = dataprep.create_padded_combined_text_sequences(
        new_content['combined_text'])

    # prepare title and description matrices,
    # which are one-hot encoded for the 10,000 most common words
    # to be fed in after the flatten layer (through fully connected layers)

    logger.info('One-hot encoding title sequences')

    title_onehot = dataprep.create_one_hot_matrix_for_column(
        tokenizing.load_tokenizer_from_file(
            os.path.join(DATADIR, "title_tokenizer.json")
        ),
        new_content['title'],
        num_words=10000,
    )

    logger.info('Title_onehot shape {}'.format(title_onehot.shape))

    logger.info('One-hot encoding description sequences')

    description_onehot = dataprep.create_one_hot_matrix_for_column(
        tokenizing.load_tokenizer_from_file(
            os.path.join(DATADIR, "description_tokenizer.json")
        ),
        new_content['description'],
        num_words=10000,
    )

    logger.info('Description_onehot shape %s', description_onehot.shape)

    logger.info('Producing arrays for new_content')

    data = {
        "x": combined_text_sequences_padded,
        "meta": meta,
        "title": title_onehot,
        "desc": description_onehot,
        "content_id": new_content['content_id']
    }

    dataprep.process_split('predict', (0, new_content.shape[0]), data)

    logger.info("Finished")

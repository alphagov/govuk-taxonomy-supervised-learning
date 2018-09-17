# coding: utf-8

import logging.config
import os
import warnings

import numpy as np
import pandas as pd
import yaml
import json

# warnings.filterwarnings(action='ignore', category=DataConversionWarning)


from dataprep import *

DATADIR = os.getenv('DATADIR')
SINCE_THRESHOLD = os.getenv('SINCE_THRESHOLD')
METADATA_LIST = os.getenv('METADATA_LIST').split()


if __name__ == "__main__":

    LOGGING_CONFIG = os.getenv('LOGGING_CONFIG')
    logging.config.fileConfig(LOGGING_CONFIG)
    logger = logging.getLogger('levelagnostic_dataprep')

    logger.info('Loading data')
    labelled = load_labelled(SINCE_THRESHOLD, level='agnostic')

    logger.info('Creating multilabel dataframe')
    binary_multilabel = create_binary_multilabel(labelled, taxon_code_column='taxon_code')
   
    print(binary_multilabel.columns)

    np.save(os.path.join(DATADIR, 'levelagnostic_taxon_codes.npy'), binary_multilabel.columns)
    taxon_codes = binary_multilabel.columns
    logging.info('Shape of binary_multilabel df: % {}'.format(binary_multilabel.shape))

    # ***** RESAMPLING OF MINORITY TAXONS **************
    # ****************************************************
    # - Training data = 80%
    # - Development data = 10%
    # - Test data = 10%

    size_before_resample = binary_multilabel.shape[0]
   
    size_train = int(0.8 * size_before_resample)  # train split
    logging.info('Size of train set: %s', size_train)

    size_dev = int(0.1 * size_before_resample)  # test split
    logging.info('Size of dev/test sets: %s', size_dev)

    logger.info('Upsample low support taxons')

    balanced_df, upsample_size = upsample_low_support_taxons(binary_multilabel, size_train)

    logger.info("Shape of balanced_df: {}".format(balanced_df.shape))
    
    size_train += upsample_size

    logger.info("New size of training set: {}".format(size_train))

    logger.info('Vectorizing metadata')

    meta = create_meta(balanced_df.index.get_level_values('content_id'), labelled)

    # **** TOKENIZE TEXT ********************
    # ************************************

    # Load tokenizers, fitted on both labelled and unlabelled data from file
    # created in clean_content.py

    logger.info('Tokenizing combined_text')

    combined_text_sequences_padded = create_padded_combined_text_sequences(
        balanced_df.index.get_level_values('combined_text')
    )

    # prepare title and description matrices,
    # which are one-hot encoded for the 10,000 most common words
    # to be fed in after the flatten layer (through fully connected layers)

    logging.info('One-hot encoding title sequences')

    title_onehot = create_one_hot_matrix_for_column(
        tokenizing.load_tokenizer_from_file(
            os.path.join(DATADIR, "title_tokenizer.json")
        ),
        balanced_df.index.get_level_values('title'),
        num_words=10000,
    )

    logger.info('Title_onehot shape {}'.format(title_onehot.shape))

    logger.info('One-hot encoding description sequences')

    description_onehot = create_one_hot_matrix_for_column(
        tokenizing.load_tokenizer_from_file(
            os.path.join(DATADIR, "description_tokenizer.json")
        ),
        balanced_df.index.get_level_values('description'),
        num_words=10000,
    )

    logger.info('Description_onehot shape')

    logger.info('Train/dev/test splitting')

    end_dev = size_train + size_dev

    logger.info('end_dev ={}'.format(end_dev))
    # assign the indices for separating the original (pre-sampled) data into
    # train/dev/test
    splits = [(0, size_train), (size_train, end_dev), (end_dev, balanced_df.shape[0])]
    logger.info('splits ={}'.format(splits))

    # convert columns to an array. Each row represents a content item,
    # each column an individual taxon
    binary_multilabel = balanced_df[list(balanced_df.columns)].values
    logger.info('Shape of multilabel array {}'.format(binary_multilabel.shape))
    logger.info('Example row of multilabel array {}'.format(binary_multilabel[2]))

    data = {
        "x": combined_text_sequences_padded,
        "meta": meta,
        "title": title_onehot,
        "desc": description_onehot,
        "y": sparse.csr_matrix(binary_multilabel),
        "content_id": balanced_df.index.get_level_values('content_id'),
    }

    for split, name in zip(splits, ('level_agnostic_train', 'level_agnostic_dev', 'level_agnostic_test')):
        logger.info("Generating {} split".format(name))
        process_split(
            name,
            split,
            data
        )

    logger.info('Finished')

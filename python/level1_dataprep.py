# coding: utf-8

import logging.config
import os
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from scipy import sparse
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle, resample
import yaml
import tokenizing
import json

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from dataprep import *

DATADIR = os.getenv('DATADIR')
SINCE_THRESHOLD = os.getenv('SINCE_THRESHOLD')
METADATA_LIST = os.getenv('METADATA_LIST').split()

def load_labelled_level1(SINCE_THRESHOLD):
    dataframe = pd.read_csv(
        os.path.join(DATADIR, 'labelled.csv.gz'),
        dtype=object,
        compression='gzip'
    )

    logging.info('dataframe shape={}'.format(dataframe.shape))
    print(dataframe.columns)
    # creating categorical variable for level2taxons from values
    dataframe['level1taxon'] = dataframe['level1taxon'].astype('category')

    # Add 1 because of zero-indexing to get 1-number of level2taxons as numerical targets
    dataframe['level1taxon_code'] = dataframe.level1taxon.astype('category').cat.codes + 1

    labels_index = dict(zip((dataframe['level1taxon_code']), dataframe['level1taxon']))

    with open(os.path.join(DATADIR, "level1taxon_labels_index.json"),'w') as f:
        json.dump(labels_index, f) 

    logging.info('Number of unique level1taxons: {}'.format(dataframe.level1taxon.nunique()))

    # count the number of taxons per content item into new column
    dataframe['num_taxon_per_content'] = dataframe.groupby(
        ["content_id"]
    )['content_id'].transform("count")

    dataframe['first_published_at'] = pd.to_datetime(dataframe['first_published_at'])
    dataframe.index = dataframe['first_published_at']
    
    dataframe = dataframe[dataframe['first_published_at'] >= pd.Timestamp(SINCE_THRESHOLD)].copy()

    return dataframe

if __name__ == "__main__":

    LOGGING_CONFIG = os.getenv('LOGGING_CONFIG')
    logging.config.fileConfig(LOGGING_CONFIG)
    logger = logging.getLogger('level1_dataprep')

    logger.info('Loading data')
    labelled_level1 = load_labelled_level1(SINCE_THRESHOLD)

    logger.info('Creating multilabel dataframe')
    binary_multilabel = create_binary_multilabel(labelled_level1, level=1)
    print(binary_multilabel.columns)

    np.save(os.path.join(DATADIR, 'level1_taxon_codes.npy'), binary_multilabel.columns)
    taxon_codes = binary_multilabel.columns

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

    size_train += upsample_size

    logger.info("New size of training set: {}".format(size_train))

    logger.info('Vectorizing metadata')

    meta = create_meta(balanced_df.index.get_level_values('content_id'), labelled_level1)

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
    logger.info('Example row of multilabel array {}'.format(binary_multilabel[2]))

    data = {
        "x": combined_text_sequences_padded,
        "meta": meta,
        "title": title_onehot,
        "desc": description_onehot,
        "y": sparse.csr_matrix(binary_multilabel),
        "content_id": balanced_df.index.get_level_values('content_id'),
    }

    for split, name in zip(splits, ('level1_train', 'level1_dev', 'level1_test')):
        logger.info("Generating {} split".format(name))
        process_split(
            name,
            split,
            data
        )

    logger.info('Finished')

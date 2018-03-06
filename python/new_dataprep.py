# coding: utf-8

import os
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
from scipy import sparse
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import dataprep
import tokenizing

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

DATADIR = os.getenv('DATADIR')

new_content = pd.read_csv(
    os.path.join(DATADIR, 'new_content.csv.gz'),
    dtype=object,
    compression='gzip'
)


new_content.drop(['content_purpose_document_supertype', 'content_purpose_subgroup',
                'content_purpose_supergroup', 'email_document_supertype',
                'government_document_supertype', 'navigation_document_supertype',
                'public_updated_at', 'search_user_need_document_supertype',
                'taxon_id', 'taxons', 'user_journey_document_supertype', 'updated_at'], axis=1, inplace=True)

# ******* Metadata ***************
# ********************************


def to_cat_to_hot(meta_df, var):
    """one hot encode each metavar"""
    encoder = LabelEncoder()
    metavar_cat = var + "_cat"  # get categorical codes into new column
    meta_df[metavar_cat] = encoder.fit_transform(meta_df[var])
    tf.cast(meta_df[metavar_cat], tf.float32)
    return to_categorical(meta_df[metavar_cat])


def create_meta(dataframe_column):

    # extract content_id index to df
    meta_df = pd.DataFrame(dataframe_column)
    meta_varlist = (
        'document_type',
        'first_published_at',
        'publishing_app',
        'primary_publishing_organisation'
    )

    for meta_var in meta_varlist:
        meta_df[meta_var] = meta_df['content_id'].map(
            dict(zip(new_content['content_id'], new_content[meta_var]))
        )

    # convert nans to empty strings for labelencoder types
    meta_df = meta_df.replace(np.nan, '', regex=True)

    dict_of_onehot_encodings = {}
    for metavar in meta_varlist:
        if metavar != "first_published_at":
            print(metavar)
            dict_of_onehot_encodings[metavar] = to_cat_to_hot(meta_df, metavar)

    # First_published_at:
    # Convert to timestamp, then scale between 0 and 1 so same weight as binary vars
    meta_df['first_published_at'] = pd.to_datetime(meta_df['first_published_at'])
    first_published = np.array(
        meta_df['first_published_at']
    ).reshape(
        meta_df['first_published_at'].shape[0],
        1
    )

    scaler = MinMaxScaler()
    first_published_scaled = scaler.fit_transform(first_published)

    last_year = np.where(
        (
            (np.datetime64('today', 'D') - first_published).astype('timedelta64[Y]')
            <
            np.timedelta64(1, 'Y')
        ),
        1,
        0
    )

    last_2years = np.where(
        (
            (np.datetime64('today', 'D') - first_published).astype('timedelta64[Y]')
            <
            np.timedelta64(2, 'Y')
        ),
        1,
        0
    )

    last_5years = np.where(
        (
            (np.datetime64('today', 'D') - first_published).astype('timedelta64[Y]')
            <
            np.timedelta64(5, 'Y')
        ),
        1,
        0
    )

    olderthan5 = np.where(
        (
            (np.datetime64('today', 'D') - first_published).astype('timedelta64[Y]')
            >
            np.timedelta64(5, 'Y')
        ),
        1,
        0
    )

    meta_np = np.concatenate(
        (dict_of_onehot_encodings['document_type'],
         dict_of_onehot_encodings['primary_publishing_organisation'],
         dict_of_onehot_encodings['publishing_app'],
         first_published_scaled,
         last_year,
         last_2years,
         last_5years,
         olderthan5),
        axis=1
    )

    return sparse.csr_matrix(meta_np)


meta = create_meta(new_content['content_id'])

# **** TOKENIZE TEXT ********************
# ************************************

# Load tokenizers, fitted on both labelled and unlabelled data from file
# created in clean_content.py


combined_text_sequences_padded = dataprep.create_passed_combined_text_sequences()

# prepare title and description matrices, 
# which are one-hot encoded for the 10,000 most common words
# to be fed in after the flatten layer (through fully connected layers)

print('one-hot encoding title sequences')

title_onehot = dataprep.create_one_hot_matrix_for_column(
    tokenizing.load_tokenizer_from_file(
        os.path.join(DATADIR,"title_tokenizer.json")
    ),
    'title',
    num_words=10000,
)

print('title_onehot shape {}'.format(title_onehot.shape))

print('one-hot encoding description sequences')

description_onehot = dataprep.create_one_hot_matrix_for_column(
    tokenizing.load_tokenizer_from_file(
        os.path.join(DATADIR,"description_tokenizer.json")
    ),
    'description',
    num_words=10000,
)

print('description_onehot shape {}'.format(description_onehot.shape))


print('producing arrays for new_content')


data = {
    "x": combined_text_sequences_padded,
    "meta": meta,
    "title": title_onehot,
    "desc": description_onehot,
    "content_id": new_content['content_id']
}

dataprep.process_split('predict', (0, new_content.shape[0]), data)

print("Finished")

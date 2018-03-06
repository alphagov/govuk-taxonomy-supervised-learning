# coding: utf-8

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.utils import shuffle, resample
import tokenizing
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils import to_categorical
from sklearn.exceptions import DataConversionWarning
import warnings
from scipy import sparse

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

DATADIR = os.getenv('DATADIR')

new_content = pd.read_csv(
    os.path.join(DATADIR, 'new_content.csv.gz'),
    dtype=object,
    compression='gzip'
)



# **** RESHAPE data long -> wide by taxon *******
# ***********************************************

# reshape to wide per taxon and keep the combined text so indexing is consistent when splitting X from Y



# ******* Metadata ***************
# ********************************


def to_cat_to_hot(meta_df, var):
    """one hot encode each metavar"""
    encoder = LabelEncoder()
    metavar_cat = var + "_cat"  # get categorical codes into new column
    meta_df[metavar_cat] = encoder.fit_transform(meta_df[var])
    tf.cast(meta_df[metavar_cat], tf.float32)
    return to_categorical(meta_df[metavar_cat])

def create_meta(balanced_df):

    # extract content_id index to df
    meta_df = pd.DataFrame(balanced_df.index.get_level_values('content_id'))
    meta_varlist = (
        'document_type',
        'first_published_at',
        'publishing_app',
        'primary_publishing_organisation'
    )

    for meta_var in meta_varlist:
        meta_df[meta_var] = meta_df['content_id'].map(
            dict(zip(labelled_level2['content_id'], labelled_level2[meta_var]))
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

    meta = np.concatenate(
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

    return sparse.csr_matrix(meta)

meta = create_meta(balanced_df)

# **** TOKENIZE TEXT ********************
# ************************************

# Load tokenizers, fitted on both labelled and unlabelled data from file
# created in clean_content.py

def create_passed_combined_text_sequences():
    tokenizer_combined_text = tokenizing.\
        load_tokenizer_from_file(os.path.join(DATADIR, "combined_text_tokenizer.json"))

    # Prepare combined text data for input into embedding layer
    print('converting combined text to sequences')
    tokenizer_combined_text.num_words = 20000
    combined_text_sequences = tokenizer_combined_text.texts_to_sequences(
        balanced_df.index.get_level_values('combined_text')
    )

    print('padding combined text sequences')
    combined_text_sequences_padded = pad_sequences(
        combined_text_sequences,
        maxlen=1000,  # MAX_SEQUENCE_LENGTH
        padding='post', truncating='post'
    )

    return combined_text_sequences_padded

combined_text_sequences_padded = create_passed_combined_text_sequences()

def create_one_hot_matrix_for_column(
        tokenizer,
        column_name,
        num_words,
):
    tokenizer.num_words = num_words
    return sparse.csr_matrix(
        tokenizer.texts_to_matrix(
            balanced_df.index.get_level_values(column_name)
        )
    )

# prepare title and description matrices, 
# which are one-hot encoded for the 10,000 most common words
# to be fed in after the flatten layer (through fully connected layers)

print('one-hot encoding title sequences')

title_onehot = create_one_hot_matrix_for_column(
    tokenizing.load_tokenizer_from_file(
        os.path.join(DATADIR,"title_tokenizer.json")
    ),
    'title',
    num_words=10000,
)

print('title_onehot shape {}'.format(title_onehot.shape))

print('one-hot encoding description sequences')

description_onehot = create_one_hot_matrix_for_column(
    tokenizing.load_tokenizer_from_file(
        os.path.join(DATADIR,"description_tokenizer.json")
    ),
    'description',
    num_words=10000,
)

print('description_onehot shape {}'.format(description_onehot.shape))

# description_tfidf = tokenizer_description.texts_to_matrix(balanced_df.index.get_level_values('description'), 'tfidf')

# ******* TRAIN/DEV/TEST SPLIT DATA ****************
# **************************************************

# - Training data = 80%
# - Development data = 10%
# - Test data = 10%


def split(data_to_split, split_indices):
    """split data along axis=0 (rows) at indices designated in split_indices"""
    return tuple(
        data_to_split[start:end]
        for (start, end) in split_indices
    )

print('train/dev/test splitting')

# assign the indices for separating the original (pre-sampled) data into
# train/dev/test
splits = [(0, balanced_df.shape[0])]
print('splits ={}'.format(splits))

def process_split(
        split_name,
        split,
        data,
):
    start, end = split

    split_data = {
        name: df[start:end]
        for name, df in data.items()
    }

    np.savez(
        os.path.join(DATADIR,'{}_arrays.npz'.format(split_name)),
        **split_data
    )

data = {
    "x": combined_text_sequences_padded,
    "meta": meta,
    "title": title_onehot,
    "desc": description_onehot,
    "y": sparse.csr_matrix(binary_multilabel),
    "content_id": balanced_df.index.get_level_values('content_id'),
}

for split, name in zip(splits, ('predict')):
    print("Generating {} split".format(name))
    process_split(
        name,
        split,
        data
    )

print("Finished")

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

# create dictionary of taxon category code to string label for use in model evaluation
labels_index = dict(zip((labelled_level2['level2taxon_code']),
                        labelled_level2['level2taxon']))

# ***** NEW COLUMNS FREQUENCY COUNTS **********
# *********************************************

# count the number of content items per taxon into new column
labelled_level2['num_content_per_taxon'] = labelled_level2.groupby(["level2taxon"])['level2taxon'].transform("count")

print('Number of unique level2taxons: {}'.format(labelled_level2.level2taxon.nunique()))

# count the number of taxons per content item into new column
labelled_level2['num_taxon_per_content'] = labelled_level2.groupby(["content_id"])['content_id'].transform("count")


# **** RESHAPE data long -> wide by taxon *******
# ***********************************************

# reshape to wide per taxon and keep the combined text so indexing is consistent when splitting X from Y

def create_binary_multilabel(labelled_level2):
    multilabel = labelled_level2.pivot_table(
        index=[
            'content_id',
            'combined_text',
            'title',
            'description'
        ],
        columns='level2taxon_code',
        values='num_taxon_per_content'
    )

    print('labelled_level2 shape: {}'.format(labelled_level2.shape))
    print('multilabel (pivot table - no duplicates): {} '.format(multilabel.shape))

    multilabel.columns.astype('str')

    # THIS IS WHY INDEXING IS NOT ZERO-BASED convert the number_of_taxons_per_content values to 1, meaning there was an
    # entry for this taxon and this content_id, 0 otherwise
    binary_multilabel = multilabel.notnull().astype('int')

    # shuffle to ensure no order is captured in train/dev/test splits
    binary_multilabel = shuffle(binary_multilabel, random_state=0)

    # delete the 1st order column name (='level2taxon') for later calls to column names (now string numbers of each taxon)
    del binary_multilabel.columns.name

    return binary_multilabel


binary_multilabel = create_binary_multilabel(labelled_level2)

# ***** RESAMPLING OF MINORITY TAXONS **************
# ****************************************************
# - Training data = 80%
# - Development data = 10%
# - Test data = 10%

size_before_resample = binary_multilabel.shape[0]

size_train = int(0.8 * size_before_resample)  # train split
print('Size of train set:', size_train)

size_dev = int(0.1 * size_before_resample)  # test split
print('Size of dev/test sets:', size_dev)


def upsample_low_support_taxons(dataframe):
    # extract indices of training samples, which are to be upsampled

    training_indices = [dataframe.index[i][0] for i in range(0, size_train)]

    upsampled_training = pd.DataFrame()
    last_taxon = len(dataframe.columns) + 1

    for taxon in range(1, last_taxon):

        training_samples_tagged_to_taxon = dataframe[
                                               dataframe[taxon] == 1
                                               ][:size_train]

        if training_samples_tagged_to_taxon.shape[0] < 500:
            print("Taxon code:", taxon, "Taxon name:", labels_index[taxon])
            print("SMALL SUPPORT:", training_samples_tagged_to_taxon.shape[0])
            df_minority = training_samples_tagged_to_taxon
            if not df_minority.empty:
                # Upsample minority class
                print(df_minority.shape)
                df_minority_upsampled = resample(df_minority,
                                                 replace=True,  # sample with replacement
                                                 n_samples=(500),
                                                 # to match majority class, switch to max_content_freq if works
                                                 random_state=123)  # reproducible results
                print("FIRST 5 IDs:", [df_minority_upsampled.index[i][0] for i in range(0, 5)])
                # Combine majority class with upsampled minority class
                upsampled_training = pd.concat([upsampled_training, df_minority_upsampled])
                # Display new shape
                print("UPSAMPLING:", upsampled_training.shape)

    upsampled_training = shuffle(upsampled_training, random_state=0)

    print("Size of upsampled_training: {}".format(upsampled_training.shape[0]))

    balanced = pd.concat([upsampled_training, dataframe])
    balanced.astype(int)
    balanced.columns.astype(int)

    return balanced, upsampled_training.shape[0]


balanced_df, upsample_size = upsample_low_support_taxons(binary_multilabel)
size_train += upsample_size
print("New size of training set: {}".format(size_train))


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


meta = create_meta(balanced_df.index.get_level_values('content_id'))


# **** TOKENIZE TEXT ********************
# ************************************

# Load tokenizers, fitted on both labelled and unlabelled data from file
# created in clean_content.py

def create_padded_combined_text_sequences(text_data):
    tokenizer_combined_text = tokenizing. \
        load_tokenizer_from_file(os.path.join(DATADIR, "combined_text_tokenizer.json"))

    # Prepare combined text data for input into embedding layer
    print('converting combined text to sequences')
    tokenizer_combined_text.num_words = 20000
    combined_text_sequences = tokenizer_combined_text.texts_to_sequences(
        text_data
    )

    print('padding combined text sequences')
    combined_text_sequences_padded = pad_sequences(
        combined_text_sequences,
        maxlen=1000,  # MAX_SEQUENCE_LENGTH
        padding='post', truncating='post'
    )

    return combined_text_sequences_padded


combined_text_sequences_padded = create_padded_combined_text_sequences(balanced_df.index.get_level_values('combined_text'))


def create_one_hot_matrix_for_column(
        tokenizer,
        column_data,
        num_words,
):
    tokenizer.num_words = num_words
    return sparse.csr_matrix(
        tokenizer.texts_to_matrix(
            column_data
        )
    )


# prepare title and description matrices,
# which are one-hot encoded for the 10,000 most common words
# to be fed in after the flatten layer (through fully connected layers)

print('one-hot encoding title sequences')

title_onehot = create_one_hot_matrix_for_column(
    tokenizing.load_tokenizer_from_file(
        os.path.join(DATADIR, "title_tokenizer.json")
    ),
    balanced_df.index.get_level_values('title'),
    num_words=10000,
)

print('title_onehot shape {}'.format(title_onehot.shape))

print('one-hot encoding description sequences')

description_onehot = create_one_hot_matrix_for_column(
    tokenizing.load_tokenizer_from_file(
        os.path.join(DATADIR, "description_tokenizer.json")
    ),
    balanced_df.index.get_level_values('description'),
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

end_dev = size_train + size_dev
print('end_dev ={}'.format(end_dev))
# assign the indices for separating the original (pre-sampled) data into
# train/dev/test
splits = [(0, size_train), (size_train, end_dev), (end_dev, balanced_df.shape[0])]
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

    for name, df in split_data.items():
        print("  {}: {}".format(name, df.shape))

    np.savez(
        os.path.join(DATADIR, '{}_arrays.npz'.format(split_name)),
        **split_data
    )


# convert columns to an array. Each row represents a content item,
# each column an individual taxon
binary_multilabel = balanced_df[list(balanced_df.columns)].values
print('Example row of multilabel array {}'.format(binary_multilabel[2]))

data = {
    "x": combined_text_sequences_padded,
    "meta": meta,
    "title": title_onehot,
    "desc": description_onehot,
    "y": sparse.csr_matrix(binary_multilabel),
    "content_id": balanced_df.index.get_level_values('content_id'),
}

for split, name in zip(splits, ('train', 'dev', 'test')):
    print("Generating {} split".format(name))
    process_split(
        name,
        split,
        data
    )

print("Finished")


def load_labelled_level2():
    labelled_level2 = pd.read_csv(
        os.path.join(DATADIR, 'labelled_level2.csv.gz'),
        dtype=object,
        compression='gzip'
    )

    # Create World taxon in case any items not identified
    # through doc type in clean_content are still present
    labelled_level2.loc[labelled_level2['level1taxon'] == 'World', 'level2taxon'] = 'world_level1'

    # **** TAXONS TO CATEGORICAL -> DICT **********
    # *********************************************

    # creating categorical variable for level2taxons from values
    labelled_level2['level2taxon'] = labelled_level2['level2taxon'].astype('category')

    # Add 1 because of zero-indexing to get 1-number of level2taxons as numerical targets
    labelled_level2['level2taxon_code'] = labelled_level2.level2taxon.astype('category').cat.codes + 1

    return labelled_level2

if __name__ == "__main__":
    labelled_level2 = load_labelled_level2()

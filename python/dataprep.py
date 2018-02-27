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

DATADIR = os.getenv('DATADIR')

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

multilabel = (labelled_level2.pivot_table(index=['content_id',
                                                 'combined_text',
                                                 'title',
                                                 'description'
                                                 ],
                                          columns='level2taxon_code',
                                          values='num_taxon_per_content')
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

# ***** RESAMPLING OF MINORITY TAXONS **************
# ****************************************************
# - Training data = 80%
# - Development data = 10%
# - Test data = 10%

size_bef_resample = binary_multilabel.shape[0]

size_train = int(0.8 * size_bef_resample)  # train split
print('Size of train set:', size_train)

size_dev = int(0.1 * size_bef_resample)  # test split
print('Size of dev/test sets:', size_dev)

# extract indices of training samples, which are to be upsampled

training_indices = [binary_multilabel.index[i][0] for i in range(0, size_train)]

upsampled_training = pd.DataFrame()
last_taxon = len(binary_multilabel.columns) + 1

for taxon in range(1, last_taxon):
    num_samples = binary_multilabel[binary_multilabel[taxon] == 1].shape[0]
    if num_samples < 500:
        print("Taxon code:", taxon, "Taxon name:", labels_index[taxon])
        print("SMALL SUPPORT:", num_samples)
        df_minority = binary_multilabel[binary_multilabel[taxon] == 1].loc[training_indices]
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

balanced_df = pd.concat([binary_multilabel, upsampled_training])

# ********** CREATE Y ARRAY **************
# ****************************************

balanced_df.astype(int)
balanced_df.columns.astype(int)
# convert columns to an array. Each row represents a content item, each column an individual taxon
binary_multilabel = balanced_df[list(balanced_df.columns)].values
print('Example row of multilabel array {}'.format(binary_multilabel[2]))

# ******* Metadata ***************
# ********************************

# extract content_id index to df
meta_df = pd.DataFrame(balanced_df.index.get_level_values('content_id'))
meta_varlist = ['document_type',
                'first_published_at',
                'publishing_app',
                'primary_publishing_organisation']

for meta_var in meta_varlist:
    meta_df[meta_var] = meta_df['content_id'].map(
        dict(zip(labelled_level2['content_id'], labelled_level2[meta_var])))

# convert nans to empty strings for labelencoder types
meta_df = meta_df.replace(np.nan, '', regex=True)


def to_cat_to_hot(var):
    """one hot encode each metavar"""
    encoder = LabelEncoder()
    metavar_cat = var + "_cat"  # get categorical codes into new column
    meta_df[metavar_cat] = encoder.fit_transform(meta_df[var])
    tf.cast(meta_df[metavar_cat], tf.float32)
    return to_categorical(meta_df[metavar_cat])


dict_of_onehot_encodings = {}
for metavar in meta_varlist:
    if metavar != "first_published_at":
        print(metavar)
        dict_of_onehot_encodings[metavar] = to_cat_to_hot(metavar)

# First_published_at:
# Convert to timestamp, then scale between 0 and 1 so same weight as binary vars
meta_df['first_published_at'] = pd.to_datetime(meta_df['first_published_at'])
first_published = np.array(meta_df['first_published_at']).reshape(meta_df['first_published_at'].shape[0], 1)

scaler = MinMaxScaler()
first_published_scaled = scaler.fit_transform(first_published)

last_year = np.where(
    (np.datetime64('today', 'D') - first_published).astype('timedelta64[Y]')
    < np.timedelta64(1, 'Y'), 1, 0)

last_2years = np.where(
    (np.datetime64('today', 'D') - first_published).astype('timedelta64[Y]')
    < np.timedelta64(2, 'Y'), 1, 0)

last_5years = np.where(
    (np.datetime64('today', 'D') - first_published).astype('timedelta64[Y]')
    < np.timedelta64(5, 'Y'), 1, 0)

olderthan5 = np.where(
    (np.datetime64('today', 'D') - first_published).astype('timedelta64[Y]')
    > np.timedelta64(5, 'Y'), 1, 0)

meta = np.concatenate((dict_of_onehot_encodings['document_type'],
                       dict_of_onehot_encodings['primary_publishing_organisation'],
                       dict_of_onehot_encodings['publishing_app'],
                       first_published_scaled,
                       last_year,
                       last_2years,
                       last_5years,
                       olderthan5),
                      axis=1)

# **** TOKENIZE TEXT ********************
# ************************************

# Load tokenizers, fitted on both labelled and unlabelled data from file
# created in clean_content.py

tokenizer_combined_text = tokenizing.\
    load_tokenizer_from_file(os.path.join(DATADIR, "combined_text_tokenizer.json"))

tokenizer_title = tokenizing.\
    load_tokenizer_from_file(os.path.join(DATADIR,"title_tokenizer.json"))

tokenizer_description = tokenizing.\
    load_tokenizer_from_file(os.path.join(DATADIR, "description_tokenizer.json"))

# Prepare combined text data for input into embedding layer

combined_text_sequences = tokenizer_combined_text.texts_to_sequences(
    balanced_df.index.get_level_values('combined_text')
)

combined_text_sequences_padded = pad_sequences(
    combined_text_sequences,
    maxlen=1000,  # MAX_SEQUENCE_LENGTH
    padding='post', truncating='post'
)

# prepare title and description matrices, 
# which are one-hot encoded for the 10,000 most common words
# to be fed in after the flatten layer (through fully connected layers)

title_sequences = tokenizer_title.texts_to_sequences(
    balanced_df.index.get_level_values('title')
)

title_onehot = tokenizer_title.sequences_to_matrix(title_sequences)

description_sequences = tokenizer_description.texts_to_sequences(
    balanced_df.index.get_level_values('description')
)

description_onehot = tokenizer_description.sequences_to_matrix(description_sequences)

# ******* TRAIN/DEV/TEST SPLIT DATA ****************
# **************************************************

# - Training data = 80%
# - Development data = 10%
# - Test data = 10%

total_size = balanced_df.shape[0]
end_dev = size_train + size_dev

splits = [(0, size_train), (size_train, end_dev), (end_dev, size_bef_resample)]
re_split = [(size_bef_resample, total_size)]


def split(data_to_split, split_indices):
    """split data along axis=0 (rows) at indices designated in split_indices"""
    list_of_split_data_subsets = []
    for (start, end) in split_indices:
        list_of_split_data_subsets.append(data_to_split[start:end])
    return tuple(list_of_split_data_subsets)


x_train, x_dev, x_test = split(combined_text_sequences_padded, splits)
x_resampled = split(combined_text_sequences_padded, re_split)[0]

meta_train, meta_dev, meta_test = split(meta, splits)
meta_resampled = split(meta, re_split)[0]
meta_train = np.concatenate([meta_train, meta_resampled], axis=0)

title_train, title_dev, title_test = split(title_onehot, splits)
title_resampled = split(title_onehot, re_split)[0]
title_train = np.concatenate([title_train, title_resampled], axis=0)

desc_train, desc_dev, desc_test = split(description_onehot, splits)
desc_resampled = split(description_onehot, re_split)[0]
desc_train = np.concatenate([desc_train, desc_resampled], axis=0)

y_train, y_dev, y_test = split(binary_multilabel, splits)
y_resampled = split(binary_multilabel, re_split)[0]
y_train = np.concatenate([y_train, y_resampled], axis=0)

np.savez_compressed(os.path.join(DATADIR,'train_arrays.npz'),
                    x=x_train,
                    meta=meta_train,
                    title=title_train,
                    desc=desc_train,
                    y=y_train)
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

size_before_resample = binary_multilabel.shape[0]

size_train = int(0.8 * size_before_resample)  # train split
print('Size of train set:', size_train)

size_dev = int(0.1 * size_before_resample)  # test split
print('Size of dev/test sets:', size_dev)

# extract indices of training samples, which are to be upsampled

training_indices = [binary_multilabel.index[i][0] for i in range(0, size_train)]

upsampled_training = pd.DataFrame()
last_taxon = len(binary_multilabel.columns) + 1

for taxon in range(1, last_taxon):
    training_samples_tagged_to_taxon = binary_multilabel[
        binary_multilabel[taxon] == 1
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
print('loading tokenizers')
tokenizer_combined_text = tokenizing.\
    load_tokenizer_from_file(os.path.join(DATADIR, "combined_text_tokenizer.json"))

tokenizer_title = tokenizing.\
    load_tokenizer_from_file(os.path.join(DATADIR,"title_tokenizer.json"))

tokenizer_description = tokenizing.\
    load_tokenizer_from_file(os.path.join(DATADIR, "description_tokenizer.json"))

# Prepare combined text data for input into embedding layer
print('converting combined text to sequences')
tokenizer_combined_text.num_words = 1000
combined_text_sequences = tokenizer_combined_text.texts_to_sequences(
    balanced_df.index.get_level_values('combined_text')
)

print('padding combined text sequences')
combined_text_sequences_padded = pad_sequences(
    combined_text_sequences,
    maxlen=1000,  # MAX_SEQUENCE_LENGTH
    padding='post', truncating='post'
)

# prepare title and description matrices, 
# which are one-hot encoded for the 10,000 most common words
# to be fed in after the flatten layer (through fully connected layers)

print('one-hot encoding title sequences')
tokenizer_title.num_words = 10000
title_onehot = tokenizer_title.texts_to_matrix(
    balanced_df.index.get_level_values('title')
)

print('title_onehot shape {}'.format(title_onehot.shape))

print('one-hot encoding description sequences')
tokenizer_description.num_words = 10000
description_onehot = tokenizer_description.texts_to_matrix(
    balanced_df.index.get_level_values('description')
)

print('description_onehot shape {}'.format(description_onehot.shape))

# ******* TRAIN/DEV/TEST SPLIT DATA ****************
# **************************************************

# - Training data = 80%
# - Development data = 10%
# - Test data = 10%

print('train/dev/test splitting')
size_after_resample = balanced_df.shape[0]
print('size_after_resmaple ={}'.format(size_after_resample))
end_dev = size_train + size_dev
print('end_dev ={}'.format(end_dev))
# assign the indices for separating the original (pre-sampled) data into
# train/dev/test
splits = [(0, size_train), (size_train, end_dev), (end_dev, size_before_resample)]
print('splits ={}'.format(splits))
# assign the indices for separating out the resampled training data
resampled_split = [(size_before_resample, size_after_resample)]
print('resampled_split ={}'.format(resampled_split))

def split(data_to_split, split_indices):
    """split data along axis=0 (rows) at indices designated in split_indices"""
    return tuple(
        data_to_split[start:end]
        for (start, end) in split_indices
    )

print('extract combined text arrays')
# extract arrays as subsets of original text data
x_train, x_dev, x_test = split(combined_text_sequences_padded, splits)
# extract array of all resampled training text data
x_resampled = split(combined_text_sequences_padded, resampled_split)[0]
# append resampled data to original training subset
x_train = np.concatenate([x_train, x_resampled], axis=0)

print('extract metadata arrays')
meta_train, meta_dev, meta_test = split(meta, splits)
meta_resampled = split(meta, resampled_split)[0]
meta_train = np.concatenate([meta_train, meta_resampled], axis=0)

print('extract title arrays')
title_train, title_dev, title_test = split(title_onehot, splits)
title_resampled = split(title_onehot, resampled_split)[0]
title_train = np.concatenate([title_train, title_resampled], axis=0)

print('extract description arrays')
desc_train, desc_dev, desc_test = split(description_onehot, splits)
desc_resampled = split(description_onehot, resampled_split)[0]
desc_train = np.concatenate([desc_train, desc_resampled], axis=0)

print('extract Y arrays')
y_train, y_dev, y_test = split(binary_multilabel, splits)
y_resampled = split(binary_multilabel, resampled_split)[0]
y_train = np.concatenate([y_train, y_resampled], axis=0)

print('convert to sparse matrices')
x_train = sparse.csr_matrix(x_train)
meta_train = sparse.csr_matrix(meta_train)
title_train = sparse.csr_matrix(title_train)
description_train = sparse.csr_matrix(desc_train)
y_train = sparse.csr_matrix(y_train)

x_dev = sparse.csr_matrix(x_dev)
meta_dev = sparse.csr_matrix(meta_dev)
title_dev = sparse.csr_matrix(title_dev)
description_dev = sparse.csr_matrix(desc_dev)
y_dev = sparse.csr_matrix(y_dev)

x_test = sparse.csr_matrix(x_test)
meta_test = sparse.csr_matrix(meta_test)
title_test = sparse.csr_matrix(title_test)
description_test = sparse.csr_matrix(desc_test)
y_test = sparse.csr_matrix(y_test)

print('saving train arrays')
np.savez(os.path.join(DATADIR,'train_arrays.npz'),
                    x=x_train,
                    meta=meta_train,
                    title=title_train,
                    desc=desc_train,
                    y=y_train)

print('saving dev arrays')
np.savez(os.path.join(DATADIR,'dev_arrays.npz'),
                    x=x_dev,
                    meta=meta_dev,
                    title=title_dev,
                    desc=desc_dev,
                    y=y_dev)

print('saving test arrays')
np.savez(os.path.join(DATADIR,'test_arrays.npz'),
                    x=x_test,
                    meta=meta_test,
                    title=title_test,
                    desc=desc_test,
                    y=y_test)

id_train, id_dev, id_test = split(meta_df['content_id'], splits)

print('saving content_id arrays')
np.savez(os.path.join(DATADIR,'content_id_arrays.npz'), train=id_train, dev=id_dev, test=id_test)

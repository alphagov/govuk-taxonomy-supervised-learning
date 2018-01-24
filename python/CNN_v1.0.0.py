# coding: utf-8
"""
Convolutional NN to classify govuk content to level2 taxons

Based on: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
"""

import os
import logging
import logging.config
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.callbacks import TensorBoard, Callback
from keras.losses import binary_crossentropy
from keras.layers import (Embedding, Input, Dense, 
                          Conv1D, MaxPooling1D, Flatten)
from sklearn.metrics import (f1_score, precision_score, recall_score)
from sklearn.metrics import (precision_recall_fscore_support)
import tensorflow as tf
import pandas as pd
import numpy as np
from pipeline_functions import write_csv
from weightedbinarycrossentropy import WeightedBinaryCrossEntropy
from model_utils import f1

# Get environmental vars from systems

LOGGING_CONFIG = os.getenv('LOGGING_CONFIG')
DATADIR = os.getenv('DATADIR')
DATAFILE = os.getenv('DATAFILE')

# Model hyperparameters

MAX_SEQUENCE_LENGTH = int(os.environ.get('MAX_SEQUENCE_LENGTH'))
EMBEDDING_DIM = int(os.environ.get('EMBEDDING_DIM'))
P_THRESHOLD = float(os.environ.get('P_THRESHOLD'))
POS_RATIO = float(os.environ.get('POS_RATIO'))
NUM_WORDS = int(os.environ.get('NUM_WORDS'))
EPOCHS = int(os.environ.get('EPOCHS'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))

PREDICTION_PROBA = float(os.environ.get('PREDICTION_PROBA'))
# Set up logging

logging.config.fileConfig(LOGGING_CONFIG)
logger = logging.getLogger('CNN_v1.0.0')

logger.info('---- Model hyperparameters ----')
logger.info('')
logger.info('MAX_SEQUENCE_LENGTH: %s', MAX_SEQUENCE_LENGTH)
logger.info('EMBEDDING_DIM:       %s', EMBEDDING_DIM)
logger.info('P_THRESHOLD:         %s', P_THRESHOLD)
logger.info('NUM_WORDS:           %s', NUM_WORDS)
logger.info('EPOCHS:              %s', EPOCHS)
logger.info('BATCH_SIZE:          %s', BATCH_SIZE)
logger.info('')
logger.info('---- Other parameters ----')
logger.info('')
logger.info('PREDICTION_PROBA:   %s', PREDICTION_PROBA)
logger.info('LOGGING_CONFIG:     %s', LOGGING_CONFIG)
logger.info('DATADIR:            %s', DATADIR)
logger.info('DATAFILE:           %s', DATAFILE)
logger.info('')
logger.info('--------------------------')
logger.info('')
# Read in content items tagged to level 2 taxons or lower in the topic taxonomy

logger.info('Loading data from %s', DATAFILE)

labelled_level2 = pd.read_csv(
    os.path.join(DATADIR, DATAFILE),
    dtype=object,
    compression='gzip'
    )

logger.info('input data has shape %s:', labelled_level2.shape)

# Collapse World level2taxons

labelled_level2.loc[labelled_level2['level1taxon'] == 'World', 'level2taxon'] = 'world_level1'

labelled_level2['level2taxon'] = labelled_level2['level2taxon'].astype('category')

# Avoid zero-indexing numeric category codes (this is explained later...)
# NOTE: Probably a better way than adding 1 to codes here.

labels = labelled_level2['level2taxon'].cat.codes + 1

# Create dictionary of taxon code to string label for use in model evaluation

labels_index = dict(zip((labels), labelled_level2['level2taxon']))

logger.debug('Number of labels extracted from %s: %s.', 
        'labelled_level2.csv.gz', len(labels_index))

# Create targets by reshaping to get columns for each level2taxon

# TODO: Probably don't need to create level2_reduced here

level2_reduced = labelled_level2[['content_id', 'level2taxon', 'combined_text']].copy()

# How many level2taxons are there?

logger.debug('Unique level2 taxons: %s.', level2_reduced.level2taxon.nunique())

# Count the number of taxons per content item into new column
level2_reduced['num_taxon_per_content'] = level2_reduced.groupby(["content_id"])['content_id'].transform("count")

# Add 1 because of zero-indexing to get 1-number of level2taxons as
# numerical targets

level2_reduced['level2taxon_code'] = level2_reduced.level2taxon.astype('category').cat.codes + 1

# Reshape to wide per taxon and keep the combined text so indexing is
# consistent when splitting X from Y

multilabel = (level2_reduced.pivot_table(
    index=['content_id', 'combined_text'], columns='level2taxon_code',
    values='num_taxon_per_content')
             )

logger.debug('Number of unique level2 taxons: %s', level2_reduced.level2taxon.nunique())
logger.debug('Level2_reduced shape: %s', level2_reduced.shape)
logger.debug('Pivot table shape (no duplicates): %s ', multilabel.shape)

"""
THIS IS WHY INDEXING IS NOT ZERO-BASED
Convert the number_of_taxons_per_content values to 1, meaning there was an entry
for this taxon and this content_id, 0 otherwise

"""

# Convert labels to binary

binary_multilabel = multilabel.notnull().astype('int')

# Will convert columns to an array of shape

logger.debug('Shape of multilabel array before train/dev/test split: %s',
        binary_multilabel[list(binary_multilabel.columns)].values.shape)

# Convert columns to an array. Each row represents a content item, each column
# an individual taxon.

binary_multilabel = binary_multilabel[list(binary_multilabel.columns)].values

logger.debug('Example row of multilabel array: %s', binary_multilabel[2])
logger.debug('Type of binary_multilabel: %s', type(binary_multilabel))

# Create language data/X

"""
Format our text samples and labels into tensors that can be fed into
a neural  network. To do this, we will rely on Keras utilities
keras.preprocessing.text.Tokenizer and keras.preprocessing.sequence.pad_sequences.
"""

# The pivot table has two indices

logger.debug('Printing indicies of multilabel target: %s', multilabel.index.names)

# Extract combined text index to array

texts = multilabel.index.get_level_values('combined_text')
logger.debug('Shape of texts variable: %s', texts.shape)

# Instantiate keras tokenizer

tokenizer = Tokenizer(num_words=NUM_WORDS) 

# Fit tokenizer to text data

tokenizer.fit_on_texts(texts)

# Tokenise texts to create a list of word indexes, where the word of rank i
# in the dataset (starting at 1) has index i

sequences = tokenizer.texts_to_sequences(texts)

# Create dictionary mapping words (str) to their rank/index (int).

logger.debug('There are %s unique tokens in texts', len(tokenizer.word_index))

# Pad sequences to MAX_SEQUENCE_LENGTH, as texts are non-standard length

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

logger.debug('Shape of label tensor: %s', binary_multilabel.shape)
logger.debug('Shape of data tensor: %s', data.shape)

indices = np.arange(data.shape[0])
labels = binary_multilabel[indices]

# Shuffle data and standardise indices

x_train, y_train, x_dev, y_dev, x_test, y_test = shuffle_split(data, labels, logger, seed=0, split={ "train": 0.8, "dev" : 0.1, "test": 0.1})
# Prepare embedding layer

embedding_layer = Embedding(
    len(tokenizer.word_index) + 1,
    EMBEDDING_DIM, 
    input_length=MAX_SEQUENCE_LENGTH
    )

# ### Custom loss function


y_true_arr = np.array([0,1,0,1], dtype="float32")
y_pred_arr = np.array([0,0,1,1], dtype="float32")
y_true = tf.constant(y_true_arr)
y_pred = tf.constant(y_pred_arr)

with tf.Session().as_default():
    logger.debug(WeightedBinaryCrossEntropy(0.5)(y_true, y_pred).eval())
    logger.debug(binary_crossentropy(y_true, y_pred).eval())

# Define model architecture

NB_CLASSES = y_train.shape[1]
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32') #MAX_SEQUENCE_LENGTH
embedded_sequences = embedding_layer(sequence_input)

x = Conv1D(128, 5, activation='relu', name = 'conv0')(embedded_sequences)

x = MaxPooling1D(5, name = 'max_pool0')(x)

x = Conv1D(128, 5, activation='relu', name = 'conv1')(x)

x = MaxPooling1D(5 , name = 'max_pool1')(x)

x = Conv1D(128, 5, activation='relu', name = 'conv2')(x)

x = MaxPooling1D(35, name = 'global_max_pool')(x)  # global max pooling

x = Flatten()(x) #reduce dimensions from 3 to 2; convert to vector + FULLYCONNECTED

x = Dense(128, activation='relu')(x)

x = Dense(NB_CLASSES, activation='sigmoid', name = 'fully_connected')(x)

model = Model(sequence_input, x)

logger.info('Model sequence input:\n%s', sequence_input)

# Compile model

# NOTE that the model only reports f1 here at present (which is a departure
# from the v1.0.0. notebook.

model.compile(loss=WeightedBinaryCrossEntropy(POS_RATIO),
              optimizer='rmsprop',
              metrics=['binary_accuracy', f1])


logger.debug('Model summary: %s', model.summary())

# Tensorboard callbacks /metrics /monitor training

tb = TensorBoard(
    log_dir='./learn_embedding_logs', histogram_freq=1,
    write_graph=True, write_images=False
    )

# In[37]:


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
 
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]

        self.val_f1s.append(f1_score(val_targ, val_predict, average='micro'))
        self.val_recalls.append(recall_score(val_targ, val_predict))
        self.val_precisions.append(precision_score(val_targ, val_predict))

        f1 = f1_score(val_targ, val_predict, average='micro')
        precision = precision_score(val_targ, val_predict),
        recall = recall_score(val_targ, val_predict)

        logger.info("Metrics: - val_f1: %s — val_precision: %s — val_recall %s", 
                fi, precision, recall)
        return

metrics = Metrics()

# Train model

# NOTE:  Tensorboard callback is disabled to reduce model run time from
# approx 3 horus to 17 minutes

model.fit(
    x_train, y_train, 
    validation_data=(x_val, y_val), 
<<<<<<< HEAD
    epochs=EPOCHS, batch_size=BATCH_SIZE
=======
    epochs=10, batch_size=128
>>>>>>> ba9fc2b... Add logging and tidy comments
)

# Evaluate model

# Training metrics

y_prob = model.predict(x_train)

logger.debug(y_prob.shape)

y_pred = y_prob.copy()
y_pred[y_pred>P_THRESHOLD] = 1
y_pred[y_pred<P_THRESHOLD] = 0

logger.info('TRAINING F1 (micro):\n\n%s', f1_score(y_train, y_pred, average='micro'))

# Return scores for each class with average=None

logger.debug('TRAINING F1 (for each class):\n\n%s,', precision_recall_fscore_support(y_train, y_pred, average=None, sample_weight=None))

# Validation metrics

y_pred_val = model.predict(x_val)

# Use P_THRESHOLD to choose predicted class

y_pred_val[y_pred_val>=P_THRESHOLD] = 1
y_pred_val[y_pred_val<P_THRESHOLD] = 0

# average= None, the scores for each class are returned.

logger.debug('DEVELOPMENT F1 (for each class):\n\n%s,'precision_recall_fscore_support(y_val, y_pred_val, average=None, sample_weight=None))

# Calculate globally by counting the total true positives, false negatives 
# and false positives.

logger.info('DEVELOPMENT F1 (micro): %s', precision_recall_fscore_support(y_val, y_pred_val, average='micro', sample_weight=None))

# ## Tag unlabelled content

def get_predictions(new_texts, df, level1taxon=False):
    """
    Process data for model input
    """
    # Yield one sequence per input tex
    new_sequences = tokenizer.texts_to_sequences(new_texts)

    new_word_index = tokenizer.word_index
    logger.debug('Found %s unique tokens.', len(new_word_index))

    x_new = pad_sequences(new_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    logger.debug('Shape of untagged tensor: %s', x_new.shape)

    # predict tag for untagged data

    y_pred_new = model.predict(x_new)

    # Get model output into pandas & get a column to track index for later
    # merge

    y_pred_new = pd.DataFrame(y_pred_new)
    y_pred_new['index_col'] = y_pred_new.index

    # Make long by taxon so easier to filter rows and examine effect of 
    #p_threshold
    
    y_pred_new = pd.melt(y_pred_new, id_vars=['index_col'],
                         var_name='level2taxon_code', value_name='probability')

    # Get taxon names
    y_pred_new['level2taxon'] = y_pred_new['level2taxon_code'].map(labels_index)

    # Get the info about the content
    if level1taxon==False:
        # Get the info about the content
        new_info = df[['base_path', 'content_id', 'title', 'description', 
                       'document_type', 'publishing_app', 'locale']]
    else:
        new_info = df[['base_path', 'content_id', 'title', 'description', 
                       'document_type', 'publishing_app', 'locale', 'level1taxon']]

    # Merge content info with taxon prediction
    
    pred_new = pd.merge(
                        left=new_info,
                        right=y_pred_new,
                        left_index=True,
                        right_on='index_col',
                        how='outer'
                       )

    # Drop the cols needed for mergingin and naming
    
    pred_new.drop(['index_col'], axis=1, inplace=True)

    # Only return rows/samples where probability is hihger than threshold

    return pred_new.loc[pred_new['probability'] > P_THRESHOLD]

# Untagged

# Read in untagged content

untagged_raw = pd.read_csv(os.path.join(DATADIR, 'untagged_content.csv.gz'), dtype=object, compression='gzip')

new_texts = untagged_raw['combined_text']

pred_untagged = get_predictions(new_texts, untagged_raw)

logger.debug('Number of unique content items: %s', pred_untagged.content_id.nunique())
logger.debug('Number of content items tagged to taxons with more than p_threshold: %s', pred_untagged.shape)

# TODO set 0.65 and 0.85 as environment vars

pred_untagged.loc[(pred_untagged['probability'] > 0.65) & (pred_untagged['probability'] < 0.85)].sort_values(by='probability', ascending=False)

write_csv(
    pred_untagged, 'pred_untagged',
    os.path.join(DATADIR, 'predictions_for_untagged_data_trainingdatatok.csv.gz'),
    logger, compression='gzip'
    )

# Apply tokenizer to our text data

tokenizer.fit_on_texts(new_texts)

pred_untagged_refit_tok = get_predictions(new_texts, untagged_raw)

# write to csv

write_csv(
    pred_untagged_refit_tok, 'pred_untagged_refit_tok',
    os.path.join(DATADIR, 'predictions_for_untagged_data_refittok.csv.gz'),
    logger, compression='gzip'
    )

# New data (untagged + old taxons)

# old_taxons data has no combined text. This needs fixing in the data 
# pipeline before being able to use these data for predictions.

#read in untagged content
new_raw = pd.read_csv(os.path.join(DATADIR, 'new_content.csv.gz'), dtype=object, compression='gzip')

# TODO explain these!

logger.debug(new_raw.shape)
logger.debug(type(new_raw['combined_text'][0]))
logger.debug(len(new_raw[new_raw['combined_text'].isna()]))
logger.debug((new_raw.loc[(new_raw['combined_text'].isna()) & (new_raw['untagged_type'] != 'untagged')]).shape)

# Make a copy so you can edit data without needed to read in each time

new_df = new_raw.copy()
pred_new = get_predictions(new_df)

# Keep only rows where prob of taxon is greater than threshold

pred_new = pred_new.loc[pred_new['probability'] > PREDICTION_PROBA]

# Write to csv

write_csv(
    pred_new, 'pred_new',
    os.path.join(DATADIR, 'predictions_for_new_data.csv.gzip'), 
    logger, compression='gzip'
    )

# Labelled at level1only

labelled_level1 = pd.read_csv(os.path.join(DATADIR, 'labelled_level1.csv.gz'), dtype=object, compression='gzip')

level1_texts = labelled_level1['combined_text']

# Reset tokenizer to training data texts

tokenizer.fit_on_texts(texts)

pred_labelled_level1 = get_predictions(level1_texts, labelled_level1, level1taxon=True)
pred_labelled_level1.sort_values(by='probability', ascending=False)

# Write to csv

write_csv(
    pred_labelled_level1, 'pred_labelled_level1',
    os.path.join(DATADIR, 'predictions_for_level1only.csv.gz'), 
    logger, compression='gzip'
    )

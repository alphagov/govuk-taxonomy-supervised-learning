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
from utils import f1, Metrics, get_predictions, shuffle_split

# Get environmental vars from systems

LOGGING_CONFIG = os.getenv('LOGGING_CONFIG')
DATADIR = os.getenv('DATADIR')

# Model hyperparameters

MAX_SEQUENCE_LENGTH = int(os.environ.get('MAX_SEQUENCE_LENGTH'))
EMBEDDING_DIM = int(os.environ.get('EMBEDDING_DIM'))
P_THRESHOLD = float(os.environ.get('P_THRESHOLD'))
POS_RATIO = float(os.environ.get('POS_RATIO'))
NUM_WORDS = int(os.environ.get('NUM_WORDS'))
EPOCHS = int(os.environ.get('EPOCHS'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))

# Hyperparameters

"""
Intuition for POS_RATIO is that it penalises the prediction of zero for
everything, which is attractive to the model because the multilabel y
matrix is super sparse.

Increasing POS_RATIO should penalise predicting zeros more.

EMBEDDING_DIM: Keras embedding layer output_dim = Dimension of the dense embedding
P_THRESHOLD: Threshold for probability of being assigned to class
POS_RATIO: Ratio of positive to negative for each class in weighted binary
cross entropy loss function
NUM_WORDS: Keras tokenizer num_words: None or int. Maximum number of words to
work with (if set, tokenization will be restricted to the top num_words most
common words in the dataset).
"""

# Read in data
# Content items tagged to level 2 taxons or lower in the topic taxonomy

labelled_level2 = pd.read_csv(
    os.path.join(DATADIR, 'labelled_level2.csv.gz'), dtype=object, compression='gzip'
    )

# Create dictionary mapping taxon codes to string labels
# Collapse World level2taxons

labelled_level2.loc[labelled_level2['level1taxon'] == 'World', 'level2taxon'] = 'world_level1'

# Creating categorical variable for level2taxons from values
labelled_level2['level2taxon'] = labelled_level2['level2taxon'].astype('category')

# Get the category numeric values (codes) and avoid zero-indexing
labels = labelled_level2['level2taxon'].cat.codes + 1

# Create dictionary of taxon category code to string label for use in model evaluation
labels_index = dict(zip((labels), labelled_level2['level2taxon']))

logger.debug('Number of labels extracted from %s: %s.', 
        'labelled_level2.csv.gz', len(labels_index))

# Create target/Y
"""
NOTE: when using the categorical_crossentropy loss, your targets should be
in categorical format (e.g. if you have 10 classes, the target for each
sample should be a 10-dimensional vector that is all-zeros expect for a 1
at the index corresponding to the class of the sample).

In multilabel learning, the joint set of binary classification tasks is
expressed with label binary indicator array: each sample is one row of a
2d array of shape (n_samples, n_classes) with binary values:

* The one, i.e. the non zero elements, corresponds to the subset of labels.
* An array such as np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]]) represents
label 0 in the first sample, labels 1 and 2 in the second sample, and no
labels in the third sample.

Producing multilabel data as a list of sets of labels may be more intuitive.
"""

# First reshape wide to get columns for each level2taxon and row
# number = number unique urls

# TODO clarify this comment:  Get a smaller copy of data for pivoting ease
# (think you can work from full data actually and other cols get droopedauto)

level2_reduced = labelled_level2[['content_id', 'level2taxon', 'combined_text']].copy()

# How many level2taxons are there?

logger.debug('Number of unique level2 taxons: %s.', level2_reduced.level2taxon.nunique())

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

logger.debug('Number of unique level2 taxons: %s.', level2_reduced.level2taxon.nunique())
logger.debug('Level2_reduced shape: %s', level2_reduced.shape)
logger.debug('Pivot table shape (no duplicates): %s ', multilabel.shape)

"""
THIS IS WHY INDEXING IS NOT ZERO-BASED
Convert the number_of_taxons_per_content values to 1, meaning there was an entry
for this taxon and this content_id, 0 otherwise

"""
binary_multilabel = multilabel.notnull().astype('int')

# Will convert columns to an array of shape

logger.debug('Shape of Y multilabel array before train/val/test split: %s',
        binary_multilabel[list(binary_multilabel.columns)].values.shape)

# Convert columns to an array. Each row represents a content item, each column
# an individual taxon

binary_multilabel = binary_multilabel[list(binary_multilabel.columns)].values

logger.debug('Shape of Y multilabel array before train/val/test split: %s',
             binary_multilabel[list(binary_multilabel.columns)].values.shape)

logger.debug('Example row of multilabel array: %s', binary_multilabel[2])

# TODO move to assert
logger.debug('Type of binary_multilabel: %s', type(binary_multilabel))

"""
Create language data/X

Format our text samples and labels into tensors that can be fed into
a neural  network. To do this, we will rely on Keras utilities
keras.preprocessing.text.Tokenizer and keras.preprocessing.sequence.pad_sequences.
"""
# The pivot table has two indices
logger.debug(multilabel.index.names)

# Extract combined text index to array
texts = multilabel.index.get_level_values('combined_text')
logger.debug(texts.shape)

"""
Tokenizer

Tokenizer = Class for vectorizing texts, or/and turning texts into sequences 
(=list of word indexes, where the word of rank i in the dataset (starting at 1) 
has index i)
"""

"""
Bag of words method

# NUM_WORDS: None or int. Maximum number of words to work with (if set, 
# tokenization will be restricted to the top num_words most common words in
# the dataset).
"""

tokenizer = Tokenizer(num_words=NUM_WORDS) 

# apply tokenizer to our text data

tokenizer.fit_on_texts(texts)

"""
List of word indexes, where the word of rank i in the dataset (starting at 1)
has index i
"""

sequences = tokenizer.texts_to_sequences(texts)

# dictionary mapping words (str) to their rank/index (int).
# Only set after fit_on_texts was called.
word_index = tokenizer.word_index

logger.debug('Found %s unique tokens', len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

logger.debug('Shape of label tensor: %s', binary_multilabel.shape)
logger.debug('Shape of data tensor: %s', data.shape)

"""
Data split

Training data = 80%
validation data = 10%
Test data = 10%
"""

# Shuffle data and standardise indices

indices = np.arange(data.shape[0])
np.random.seed(0)
np.random.shuffle(indices)

data = data[indices]
labels = binary_multilabel[indices]

# NOTE The below separation is random selection WITH replacement
# so some of the test data will be in the training set!

nb_test_samples = int(0.1 * data.shape[0])
nb_val_samples = int(0.2 * data.shape[0])
nb_training_samples = int(0.8 * data.shape[0])

logger.debug('nb_test samples: %s', nb_test_samples)
logger.debug('nb_val samples: %s', nb_val_samples)
logger.debug('nb_training samples: %s', nb_training_samples)

x_train = data[:-nb_val_samples]
y_train = labels[:-nb_val_samples]

x_val = data[-nb_val_samples:-nb_test_samples]
y_val = labels[-nb_val_samples:-nb_test_samples]

x_test = data[-nb_test_samples:]
y_test = labels[-nb_test_samples:]

logger.debug('Shape of x_train: %s', x_train.shape)
logger.debug('Shape of y_train: %s', y_train.shape)
logger.debug('Shape of x_val: %s', x_val.shape)
logger.debug('Shape of y_val: %s', y_val.shape)
logger.debug('Shape of x_val: %s', x_val.shape)
logger.debug('Shape of y_val: %s', y_val.shape)

# Check these are different arrays!
np.array_equal(y_val, y_test)

"""
Preparing the Embedding layer

An Embedding layer should be fed sequences of integers, i.e. a 2D input of shape (samples, indices). These input sequences should be padded so that they all have the same length in a batch of input data (although an Embedding layer is capable of processing sequence of heterogenous length, if you don't pass an explicit input_length argument to the layer).
 
 All that the Embedding layer does is to map the integer inputs to the vectors found at the corresponding index in the embedding matrix, i.e. the sequence [1, 2] would be converted to [embeddings[1], embeddings[2]]. This means that the output of the Embedding layer will be a 3D tensor of shape (samples, sequence_length, embedding_dim).
"""

# NOTE Stopwords haven't been removed yet...

embedding_layer = Embedding(len(word_index) + 1, 
                            EMBEDDING_DIM, 
                            input_length=MAX_SEQUENCE_LENGTH)

# ### Custom loss function

class WeightedBinaryCrossEntropy(object):

    def __init__(self, pos_ratio):
        neg_ratio = 1. - pos_ratio
        # self.pos_ratio = tf.constant(pos_ratio, tf.float32)
        self.pos_ratio = pos_ratio
        # self.weights = tf.constant(neg_ratio / pos_ratio, tf.float32)
        self.weights = neg_ratio / pos_ratio
        self.__name__ = "weighted_binary_crossentropy({0})".format(pos_ratio)

    def __call__(self, y_true, y_pred):
        return self.weighted_binary_crossentropy(y_true, y_pred)

    def weighted_binary_crossentropy(self, y_true, y_pred):
            # Transform to logits
            epsilon = tf.convert_to_tensor(K.common._EPSILON, y_pred.dtype.base_dtype)
            y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
            y_pred = tf.log(y_pred / (1 - y_pred))

            cost = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, self.weights)
            return K.mean(cost * self.pos_ratio, axis=-1)
    
y_true_arr = np.array([0,1,0,1], dtype="float32")
y_pred_arr = np.array([0,0,1,1], dtype="float32")
y_true = tf.constant(y_true_arr)
y_pred = tf.constant(y_pred_arr)

with tf.Session().as_default(): 
    print(WeightedBinaryCrossEntropy(0.5)(y_true, y_pred).eval())
    print(binary_crossentropy(y_true, y_pred).eval())


# ### difficulty getting global precision/recall metrics . CAUTION interpreting monitoring metrics
# fcholltet: "Basically these are all global metrics that were approximated
# batch-wise, which is more misleading than helpful. This was mentioned in
# the docs but it's much cleaner to remove them altogether. It was a mistake
# to merge them in the first place."

# In[28]:


def mcor(y_true, y_pred):
     #matthews_correlation
     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
     y_pred_neg = 1 - y_pred_pos
 
 
     y_pos = K.round(K.clip(y_true, 0, 1))
     y_neg = 1 - y_pos
 
 
     tp = K.sum(y_pos * y_pred_pos)
     tn = K.sum(y_neg * y_pred_neg)
 
 
     fp = K.sum(y_neg * y_pred_pos)
     fn = K.sum(y_pos * y_pred_neg)
 
 
     numerator = (tp * tn - fp * fn)
     denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
 
 
     return numerator / (denominator + K.epsilon())

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))


# ## Training a 1D convnet

# ### 1. Create model

# In[29]:


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

sequence_input

# Compile model
# Note that the model only reports f1 here at present (which is a departure from
# the v1.0.0. notebook.

model.compile(loss=WeightedBinaryCrossEntropy(POS_RATIO),
              optimizer='rmsprop',
              metrics=['binary_accuracy', f1])



logger.debug(model.summary())

"""
Metric values are recorded at the end of each epoch on the training dataset.
If a validation dataset is also provided, then the metric recorded is also
calculated for the validation dataset.

All metrics are reported in verbose output and in the history object returned
from calling the fit() function. In both cases, the name of the metric
function is used as the key for the metric values. In the case of metrics
for the validation dataset, the “val_” prefix is added to the key.

You have now built a function to describe your model. To train and test this
model, there are four steps in Keras:

1. Create the model by calling the function above
2. Compile the model by calling `model.compile(...)`
3. Train the model on train data by calling `model.fit(...)`
4. Test the model on test data by calling `model.evaluate(x = ..., y = ...)`

If you want to know more about `model.compile()`, `model.fit()`, 
`model.evaluate()` and their arguments, refer to the official Keras
documentation https://keras.io/models/model/.
""" 


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

# NOTE: Disable tensorboard callback which massively increases model runtime
# from 17 minutes to 3 hours (roughly!). Add in callbacks=[tb] to replace.

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

logger.debug(f1_score(y_train, y_pred, average='micro'))

# average= None, the scores for each class are returned.
logger.debug(
        precision_recall_fscore_support(y_train, y_pred, average=None, sample_weight=None)
        )

# Validation metrics

y_pred_val = model.predict(x_val)

y_pred_val[y_pred_val>=P_THRESHOLD] = 1
y_pred_val[y_pred_val<P_THRESHOLD] = 0

# average= None, the scores for each class are returned.

logger.debug(precision_recall_fscore_support(y_val, y_pred_val, average=None, sample_weight=None))

# Calculate globally by counting the total true positives, false negatives 
# and false positives.

logger.debug(precision_recall_fscore_support(
    y_val, y_pred_val, average='micro', sample_weight=None)
     )

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

untagged_raw = pd.read_csv(os.path.join(DATADIR, 'untagged_content.csv'), dtype=object)

new_texts = untagged_raw['combined_text']

pred_untagged = get_predictions(new_texts, untagged_raw)

logger.debug('Number of unique content items: %s', pred_untagged.content_id.nunique())
logger.debug('Number of content items tagged to taxons with more than p_threshold: %s', pred_untagged.shape)

# TODO set 0.65 and 0.85 as environment vars

pred_untagged.loc[(pred_untagged['probability'] > 0.65) & (pred_untagged['probability'] < 0.85)].sort_values(by='probability', ascending=False)

# TODO Use the logging friendly class and method defined pipeline_functions.py

pred_untagged.to_csv(os.path.join(DATADIR, 'predictions_for_untagged_data_trainingdatatok.csv'), index=False)

# Apply tokenizer to our text data

tokenizer.fit_on_texts(new_texts)

pred_untagged_refit_tok = get_predictions(new_texts, untagged_raw)

# TODO Use the logging friendly class and method defined pipeline_functions.py

# write to csv
pred_untagged_refit_tok.to_csv(os.path.join(DATADIR, 'predictions_for_untagged_data_refittok.csv'), index=False)

# New data (untagged + old taxons)

# old_taxons data has no combined text. This needs fixing in the data pipeline
# before being able to use these data for predictions.

#read in untagged content
new_raw = pd.read_csv(os.path.join(DATADIR, 'new_content.csv'), dtype=object)

# TODO explain these!

logger.debug(new_raw.shape)
logger.debug(type(new_raw['combined_text'][0]))
logger.debug(len(new_raw[new_raw['combined_text'].isna()]))
logger.debug((new_raw.loc[(new_raw['combined_text'].isna()) & (new_raw['untagged_type'] != 'untagged')]).shape)

# Make a copy so you can edit data without needed to read in each time
new_df = new_raw.copy()

pred_new = get_predictions(new_df)

# TODO Set this probability in an environment var
# Keep only rows where prob of taxon > 0.5

pred_new = pred_new.loc[pred_new['probability'] > 0.5]

# TODO Use the logging friendly class and method defined pipeline_functions.py
# write to csv

pred_new.to_csv(os.path.join(DATADIR, 'predictions_for_new_data.csv'), index=False)

# Labelled at level1only

labelled_level1 = pd.read_csv(os.path.join(DATADIR, 'labelled_level1.csv'), dtype=object)

level1_texts = labelled_level1['combined_text']

# Reset tokenizer to training data texts

tokenizer.fit_on_texts(texts)

pred_labelled_level1 = get_predictions(level1_texts, labelled_level1, level1taxon=True)
pred_labelled_level1.sort_values(by='probability', ascending=False)

# TODO Use the logging friendly class and method defined pipeline_functions.py
# Write to csv

pred_labelled_level1.to_csv(os.path.join(DATADIR, 'predictions_for_level1only.csv'), index=False)

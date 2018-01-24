# coding: utf-8
"""
Helper functions for model evaluation
"""

import tensorflow as tf
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback
import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score)


class WeightedBinaryCrossEntropy(object):
    """
    Weighted Binary Cross Entropy
    """

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


def get_predictions(new_texts, df, model, labels_index, tokenizer, logger, max_sequence_length, p_threshold=0.5, level1taxon=False):
    """
    Process data for model input

    :param new_texts: <pd.DataFrame> New texts to be labelled
    :param df: <pd.DataFrame>
    :param model: Keras model object
    :param labels_index: <dict> Mapping of taxon code to string label
    :param tokenizer: <keras.preprocessing.test.Tokenizer> tokenizer object
    to be used for tokenization
    :param max_sequence_length: <int> Passed from env var MAX_SEQUENCE_LENGTH
    :param logger: <logging.getLogger()> Logging object
    :param p_threshold: <float> Passed from env var P_THRESHOLD
    :param level1taxon: <bool> Are you classifying level1taxons?
    """
    # Yield one sequence per input text

    new_sequences = tokenizer.texts_to_sequences(new_texts)
    new_word_index = tokenizer.word_index

    logger.debug('Found %s unique tokens.', len(new_word_index))

    x_new = pad_sequences(new_sequences, maxlen=max_sequence_length)

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

    subset = ['base_path', 'content_id', 'title', 'description',
              'document_type', 'publishing_app', 'locale']

    # Get the info about the content
    if level1taxon:
        new_info = df[subset.append('level1taxon')]
    else:
        new_info = df[subset]

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

    return pred_new.loc[pred_new['probability'] > p_threshold]


class Metrics(Callback):
    """
    """

    def __init__(self, logger):
        self.logger = logger
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

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

        self.logger.info("Metrics: - val_f1: %s — val_precision: %s — val_recall %s", f1, precision, recall)
        return


def shuffle_split(data, labels, logger, seed=0, split={ "train": 0.8, "dev" : 0.1, "test": 0.1}):
    """
    Perform three way split of the data:

    :param data: <np.array> input data
    :param labels: <list> target classes for classification
    :param logger: <logging.getLogger> Logging object
    :param seed: <int> random seed
    :param split: <dict> A dict of len(split)=3 describing train/dev/test
    splits.
    """

    indices = np.arange(data.shape[0])
    np.random.seed(seed)
    np.random.shuffle(indices)

    data = data[indices]

    nb_training_samples = int(split["train"] * data.shape[0])
    nb_dev_samples = int((split["dev"] + split["test"]) * data.shape[0])
    nb_test_samples = int(split["test"] * data.shape[0])

    logger.debug('nb_test samples: %s', nb_test_samples)
    logger.debug('nb_dev samples: %s', nb_dev_samples)
    logger.debug('nb_training samples: %s', nb_training_samples)

    x_train = data[:-nb_dev_samples]
    y_train = labels[:-nb_dev_samples]

    x_dev = data[-nb_dev_samples:-nb_test_samples]
    y_dev = labels[-nb_dev_samples:-nb_test_samples]

    x_test = data[-nb_test_samples:]
    y_test = labels[-nb_test_samples:]

    logger.info('Shape of x_train: %s', x_train.shape)
    logger.info('Shape of y_train: %s', y_train.shape)
    logger.info('Shape of x_dev: %s', x_dev.shape)
    logger.info('Shape of y_dev: %s', y_dev.shape)
    logger.info('Shape of x_test: %s', x_test.shape)
    logger.info('Shape of y_test: %s', y_test.shape)

    return x_train, y_train, x_dev, y_dev, x_test, y_test


def f1(y_true, y_pred):
    """
    Use Recall and precision metrics to calculate harmonic mean (f1)

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1 = 2*((precision*recall)/(precision+recall))

    return f1

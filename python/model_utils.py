# coding: utf-8
"""
Helper functions for model evaluation
"""

import tensorflow as tf
import keras.backend as K
from keras.callbacks import Callback
import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score)

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

    * Training
    * Development
    * Test

    :param data: <np.array> input data
    :param labels: <list> target classes for classification
    :param logger: <logging.getLogger> Logging object
    :param seed: <int> random seed
    :param split: <dict> A list of len(split)=3 describing train/dev/test
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

def mcor(y_true, y_pred):
    """
    Matthews' correlation
    """
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

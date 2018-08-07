import os
import tensorflow as tf
import pandas as pd
import keras.backend as K
import numpy as np

DATADIR = os.getenv('DATADIR')

class WeightedBinaryCrossEntropy(object):

    def __init__(self, pos_ratio):
        neg_ratio = 1. - pos_ratio
        #self.pos_ratio = tf.constant(pos_ratio, tf.float32)
        self.pos_ratio = pos_ratio
        #self.weights = tf.constant(neg_ratio / pos_ratio, tf.float32)
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


def f1(y_true, y_pred):
    """Use Recall  and precision metrics to calculate harmonic mean (F1 score).

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1 = 2 * ((precision * recall) / (precision + recall))

    return f1


def to_file(array, name, y_train):
    df = pd.DataFrame(data=array.tolist(), columns=[i for i in range(1, y_train.shape[1] + 1)])
    df.to_csv(os.path.join(DATADIR, name + '.csv.gz'), compression='gzip', index=False)


def get_predictions(data_to_tag, model):
    filename = data_to_tag + "_arrays.npz"
    arrays = np.load(os.path.join(DATADIR, filename))

    print('Set up arrays for new_content: {}'.format(arrays.files))
    x_predict = arrays['x']
    meta_predict = arrays['meta'].all().todense()
    title_predict = arrays['title'].all().todense()
    desc_predict = arrays['desc'].all().todense()

    print('x_arrays.shape = {}'.format(x_predict.shape))
    print('meta_arrays.shape = {}'.format(meta_predict.shape))
    print('title_arrays.shape = {}'.format(title_predict.shape))
    print('desc_arrays.shape = {}'.format(desc_predict.shape))

    print('Predict on untagged content')
    y_pred_new = model.predict([meta_predict, title_predict, desc_predict, x_predict])

    to_file(y_pred_new, data_to_tag + "_predictions")


# coding: utf-8

""" Tests for WeightedBinaryCrossEntropy
"""

import keras
import numpy as np
import tensorflow as tf
from keras.losses import binary_crossentropy
from weightedbinarycrossentropy import WeightedBinaryCrossEntropy

class TestModelUtils(object):

    def test_weightedbinarycrossentropy(self):
        """
        Test for the write_csv function
        """

        y_true_arr = np.array([0, 1, 0, 1], dtype="float32")
        y_pred_arr = np.array([0, 0, 1, 1], dtype="float32")
        y_true = tf.constant(y_true_arr)
        y_pred = tf.constant(y_pred_arr)

        with tf.Session().as_default():
            wcbe_val = WeightedBinaryCrossEntropy(0.5)(y_true, y_pred).eval()
            bc_val = binary_crossentropy(y_true, y_pred).eval()

        assert wcbe_val < bc_val

""" Tests for functions in clean_taxons.py and clean_content.py
"""
# coding: utf-8

import logging
import pandas as pd
import numpy as np
from model_utils import shuffle_split

class TestModelUtils(object):


    def setup_method(self):
        """
        Setup test conditions for subsequent method calls.
        For more info, see: https://docs.pytest.org/en/2.7.3/xunit_setup.html
        """
        self.logger = logging.getLogger('test_model_utils')


    def test_shuffle_split(self):
        """
        Test for the shuffle_split function
        """

        # Create a test dataframe

        data = np.random.randint(100, size=(100, 4))
        labels = np.random.randint(4, size=(100))

        # Create splits

        x_train, y_train, x_dev, y_dev, x_test, y_test = shuffle_split(data, labels, self.logger)

        assert x_train.shape[0] == data.shape[0] * 0.8
        assert x_dev.shape[0] == data.shape[0] * 0.1
        assert x_test.shape[0] == data.shape[0] * 0.1

        assert y_train.shape[0] == data.shape[0] * 0.8
        assert y_dev.shape[0] == data.shape[0] * 0.1
        assert y_test.shape[0] == data.shape[0] * 0.1

        assert ~np.array_equal(x_dev, x_test)
        assert ~np.array_equal(y_dev, y_test)

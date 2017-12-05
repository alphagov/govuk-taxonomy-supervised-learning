""" Tests for functions in clean_taxons.py and clean_content.py
"""
# coding: utf-8

import logging
import os
import pandas as pd
import pipeline_functions

class TestPipelineFunctions(object):


    def setup_method(self):
        """
        Setup test conditions for subsequent method calls.
        For more info, see: https://docs.pytest.org/en/2.7.3/xunit_setup.html
        """
        self.logger = logging.getLogger('test_pipeline_functions')
        self.TEST_PATH = '/tmp/test_data.csv'


    def teardown_method(self):

        os.remove(self.TEST_PATH)


    def test_write_csv(self):
        """
        Test for the write_csv function
        """

        # Create a test file

        test_data = ['test'] * 100
        test_data = pd.DataFrame(test_data)

        pipeline_functions.write_csv(test_data, 
                'test data', self.TEST_PATH, self.logger)

        assert os.path.exists(self.TEST_PATH)

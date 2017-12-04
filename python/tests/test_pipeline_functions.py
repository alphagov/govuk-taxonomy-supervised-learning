""" Tests for functions in clean_taxons.py and clean_content.py
"""
# coding: utf-8

import pytest
import pandas as pd

class TestCleanTaxons(object):

    def setup_method(self):
        """
        Setup test conditions for subsequent method calls.
        For more info, see: https://docs.pytest.org/en/2.7.3/xunit_setup.html
        """

        # Load in test data as pandas dataframe
        # Test urls are duplicated twice.

        # Note that self.urls is a dataframe so we must specify the appropriate
        # column: `url`

    def test_govukurls_deduplication(self):
        """
        Test that duplicate urls are successfully removed by the init method.
        """

        assert len(self.urlsclass.dedupurls) < len(self.urls)
        assert len(self.urlsclass.dedupurls) == len(self.urls) / 2


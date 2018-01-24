import unittest
from data_extraction import plek
from unittest.mock import patch


class TestPlek(unittest.TestCase):
    def test_dev_domain(self):
        self.assertEqual(plek.find("tests-service"), "http://tests-service.dev.gov.uk")

    def test_staging(self):
        with patch.dict('os.environ', {'GOVUK_APP_DOMAIN': 'staging.publishing.service.gov.uk'}):
            self.assertEqual(plek.find("tests-service"), "https://tests-service.staging.publishing.service.gov.uk")

    def test_defined_uri(self):
        with patch.dict('os.environ', {'PLEK_SERVICE_TEST_SERVICE_URI': 'https://tests-service.gov.uk/path'}):
            self.assertEqual(plek.find("test-service"), 'https://tests-service.gov.uk/path')

import unittest
from data_extraction.helpers import dig, slice, merge


class TestHelpers(unittest.TestCase):
    def test_slice(self):
        self.assertEqual(slice({"a": 1, "b": 2}, ["a"]), {"a": 1})

    def test_empty_dig(self):
        self.assertEqual(dig({}, "a", "b", "c"), None)

    def test_dig(self):
        self.assertEqual(dig({"a": {"b": {"c": "d"}}}, "a", "b", "c"), "d")

    def test_merge(self):
        self.assertEqual(merge({"a": 1}, {"b": 2}), {"a": 1, "b": 2})

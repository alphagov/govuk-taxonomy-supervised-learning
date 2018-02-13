import unittest
from lib.helpers import dig, slice, merge


class TestHelpers(unittest.TestCase):
    def test_slice(self):
        self.assertEqual(slice({"a": 1, "b": 2}, ["a"]), {"a": 1})

    def test_empty_dig(self):
        self.assertEqual(dig({}, "a", "b", "c"), None)

    def test_dig(self):
        self.assertEqual(dig({"a": {"b": {"c": "d"}}}, "a", "b", "c"), "d")

    def test_dig_none(self):
        self.assertEqual(dig({"a": {"b": {"c": "d"}}}, "a", "b", "q"), None)

    def test_dig_none2(self):
        self.assertEqual(dig({"a": {"b": {"c": "d"}}}, "a", "q", "b"), None)

    def test_dig_array(self):
        self.assertEqual(dig({"a": {"b": [1, 2, 3]}}, "a", "b", 1), 2)

    def test_dig_deep_array(self):
        self.assertEqual(dig({"a": {"b": [1, {'q': 'z'}, 3]}}, "a", "b", 1, 'q'), 'z')

    def test_merge(self):
        self.assertEqual(merge({"a": 1}, {"b": 2}), {"a": 1, "b": 2})

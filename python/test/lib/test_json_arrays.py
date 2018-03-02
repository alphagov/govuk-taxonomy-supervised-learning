import unittest
from lib import json_arrays
import io


class TestHelpers(unittest.TestCase):
    def test_write_json(self):
        output = io.StringIO()
        json_arrays.write_json(output, iter([{"a": 1}, {"b": 2}]))
        expected_output = """[ {"a": 1},
{"b": 2}]
"""
        self.assertEquals(output.getvalue(), expected_output)

import unittest
from lib import json_arrays
import io


class TestHelpers(unittest.TestCase):
    def test_read_json(self):
        test_string = """  [ {"a":"2", "b": [1,2]}, 
                          {"a":1, "b":2}] """

        generator = json_arrays.read_json(io.StringIO(test_string))
        self.assertDictEqual(next(generator), {'a': '2', 'b': [1, 2]})
        self.assertDictEqual(next(generator), {'a': 1, 'b': 2})


    def test_write_json(self):
        output = io.StringIO()
        json_arrays.write_json(output, iter([{"a": 1}, {"b": 2}]))
        expected_output = """[ {"a": 1},
{"b": 2}]
"""
        self.assertEquals(output.getvalue(), expected_output)

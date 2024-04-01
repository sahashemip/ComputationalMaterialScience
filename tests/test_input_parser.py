import unittest
from pathlib import Path
import numpy as np

from input_parser import InputParser

class TestInputParser(unittest.TestCase):
    '''Test the methods of InputParser calss'''
    def setUp(self):
        '''Prepare a temporary YAML file for testing.'''
        self.temp_file = Path('./input_test.yaml')
        self.invalid_key_file = Path('./invalid_test_input.yaml')

    def test_init_valid_path(self):
        '''Test initialization with a valid file path.'''
        parser = InputParser(self.temp_file)
        self.assertEqual(parser.input_file, self.temp_file)

    def test_init_invalid_path(self):
        '''Test initialization with an invalid file path raises FileNotFoundError.'''
        with self.assertRaises(FileNotFoundError):
            InputParser(Path("./non_existent_file.yaml"))

    def test_parse_raman_tensors(self):
        '''Test parsing valid Raman tensors from the input file.'''
        parser = InputParser(self.temp_file)
        raman_tensors = parser.parse_raman_tensors()
        self.assertTrue(
            np.array_equal(
                raman_tensors[0],
                np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
                )
            )

    def test_parse_propagation_vectors(self):
        '''Test parsing valid propagation vectors from the input file.'''
        parser = InputParser(self.temp_file)
        propagation_vectors = parser.parse_propagation_vectors()
        self.assertTrue(
            np.array_equal(
                propagation_vectors[0],
                np.array([1, 3, 1])
                )
            )

    def test_parse_polarization_vectors(self):
        '''Test parsing valid polarization vectors from the input file.'''
        parser = InputParser(self.temp_file)
        polarization_vectors = parser.parse_polarization_vectors()
        self.assertTrue(
            np.array_equal(
                polarization_vectors[0],
                np.array([0, 1, 0])
                )
            )

    def test_missing_raman_tensor_key_raises_keyerror(self):
        '''Test that missing 'ramantensor' key raises a KeyError.'''
        with self.assertRaises(KeyError) as context:
            parser = InputParser(self.invalid_key_file)
            parser.parse_raman_tensors()
        self.assertIn("Key 'ramantensor' not found in the input!", str(context.exception))

    def test_missing_propagation_vector_key_raises_keyerror(self):
        '''Test that missing 'propagationvector' key raises a KeyError.'''
        with self.assertRaises(KeyError) as context:
            parser = InputParser(self.invalid_key_file)
            parser.parse_propagation_vectors()
        self.assertIn("Key 'propagationvector' not found in input!", str(context.exception))

    def test_missing_polarization_vector_key_raises_keyerror(self):
        '''Test that missing 'polarizationvector' key raises a KeyError.'''
        with self.assertRaises(KeyError) as context:
            parser = InputParser(self.invalid_key_file)
            parser.parse_polarization_vectors()
        self.assertIn("Key 'polarizationvector' not found in input!", str(context.exception))


if __name__ == '__main__':
    print('Testing input_parser.py module')
    unittest.main()

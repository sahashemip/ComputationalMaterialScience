import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from polarization_orientation_Raman import InputParser, MathTools


class TestInputParser(unittest.TestCase):
    def test_read_input_successfully_read(self):
        ''' Test reading valid YAML content '''
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            yaml.dump({'key': 'value'}, f)
            temp_file = f.name
        
        inpdata = InputParser(Path(temp_file))
        data = inpdata.read_input()
        self.assertEqual(data, {'key': 'value'})
        Path(temp_file).unlink(missing_ok=True)

    def test_read_input_file_not_found(self):
        ''' Test file not found error '''
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            temp_file = f.name
        Path(temp_file).unlink()  
        with self.assertRaises(FileNotFoundError):
            InputParser(Path(temp_file))

    def test_read_input_invalid_format(self):
        ''' Test reading invalid YAML format '''
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('key: value\ninvalid_yaml: :')
            temp_file = f.name

        inpdata = InputParser(Path(temp_file))
        with self.assertRaises(yaml.YAMLError):
            inpdata.read_input()

        Path(temp_file).unlink(missing_ok=True)

    def test_parse_raman_tensors_successfully_done(self):
        '''Test parsing Raman tensors with valid data.'''
        test_data = {'ramantensor': {'R1': [[1, 2, 3],
                                            [4, 5, 6],
                                            [7, 7, 7],
                                            ]}}

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            yaml.dump(test_data, f)
            temp_file = f.name
        
        inpdata = InputParser(Path(temp_file))
        result = inpdata.parse_raman_tensors()
        
        self.assertEqual(result, [[[1, 2, 3], [4, 5, 6], [7, 7, 7]]])
        Path(temp_file).unlink(missing_ok=True)

    def test_parse_raman_tensors_key_error(self):
        '''Test parsing Raman tensors with missing key raises KeyError.'''
        invalid_key = {'ramai': {'R1': [[1, 2, 3],
                                        [4, 5, 6],
                                        [1, 1, 1],
                                        ]}}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            yaml.dump(invalid_key, f)
            temp_file = f.name

        inpdata = InputParser(Path(temp_file))
        with self.assertRaises(KeyError):
            inpdata.parse_raman_tensors()

        Path(temp_file).unlink(missing_ok=True)

    def test_parse_raman_tensors_contains_3x3_matrices(self):
        '''Test each parsed Raman tensor shape is 3x3.'''
        test_data = {'ramantensor': {'R1': [[1, 2, 3],
                                            [4, 5, 6],
                                            [8, 8, 3],
                                            ]}}

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            yaml.dump(test_data, f)
            temp_file = f.name

        inpdata = InputParser(Path(temp_file))
        result = inpdata.parse_raman_tensors()

        for element in result:
            self.assertEqual(np.array(element).shape,
                            (3, 3),
                            'Matrix is not 3x3!')

        Path(temp_file).unlink(missing_ok=True)

    def test_parse_propagation_vectors_successfully_done(self):
        '''Test parsing propagation vector with valid data'''
        valid_test = {'propagationvector': {'qs1': [3, 1, 1]}}
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            yaml.dump(valid_test, f)
            temp_file = f.name
        
        inpdata = InputParser(Path(temp_file))
        result = inpdata.parse_propagation_vectors()
        self.assertEqual(result, [[3, 1, 1]])
        
        Path(temp_file).unlink(missing_ok=True)

    def test_parse_propagation_vectors_with_invalid_key_fail(self):
        '''Test parsing propagation vecotr with invalid key'''
        invalid_test = {'propagvect': {'qs1': [3, 1, 1]}}
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            yaml.dump(invalid_test, f)
            temp_file = f.name
        
        inpdata = InputParser(Path(temp_file))
        with self.assertRaises(KeyError):
            inpdata.parse_propagation_vectors()
            
        Path(temp_file).unlink(missing_ok=True)

    def test_parse_propagation_vectors_matrix_shape_invalid(self):
        '''Test parsing propagation vector shape to be 3x1'''
        test_data = {'propagationvector': {'qs1': [3, 1, 1]}}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            yaml.dump(test_data, f)
            temp_file = f.name
            
        inpdata = InputParser(Path(temp_file))
        result = inpdata.parse_propagation_vectors()

        for element in result:
            self.assertEqual(np.array(element).shape,
                            (3,),
                            'Matrix is not 3x1!')

        Path(temp_file).unlink(missing_ok=True)

    def test_parse_polarization_vectors_successfully_done(self):
        '''Test parsing polarization vector with valid data'''
        valid_test = {'polarizationvector': {'axis1': [3, 1, 1]}}
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            yaml.dump(valid_test, f)
            temp_file = f.name
        
        inpdata = InputParser(Path(temp_file))
        result = inpdata.parse_polarization_vectors()
        self.assertEqual(result, [[3, 1, 1]])
        
        Path(temp_file).unlink(missing_ok=True)

    def test_parse_polarization_vectors_with_invalid_key_fail(self):
        '''Test parsing polarization vecotr with invalid key'''
        invalid_test = {'polarizationvect': {'qs1': [3, 1, 1]}}
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            yaml.dump(invalid_test, f)
            temp_file = f.name
        
        inpdata = InputParser(Path(temp_file))
        with self.assertRaises(KeyError):
            inpdata.parse_polarization_vectors()
            
        Path(temp_file).unlink(missing_ok=True)

    def test_parse_polarization_vectors_matrix_shape_invalid(self):
        '''Test parsing polarization vector shape to be 3x1'''
        test_data = {'polarizationvector': {'qs1': [3, 1, 1]}}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            yaml.dump(test_data, f)
            temp_file = f.name
            
        inpdata = InputParser(Path(temp_file))
        result = inpdata.parse_polarization_vectors()

        for element in result:
            self.assertEqual(np.array(element).shape,
                            (3,),
                            'Matrix is not 3x1!')

        Path(temp_file).unlink(missing_ok=True)


class TestMathTools(unittest.TestCase):
    '''Tests methods in MathTools class'''
    
    def test_get_normalized_vector_sucessfully_done(self):
        '''Test computing normal vector for valid input.'''
        inp_arr = np.array([3, 4, 5])
        normalized_arr = MathTools.get_normalized_vector(inp_arr)
        expected_result = np.array([0.42426407, 0.56568542, 0.70710678,])
        
        self.assertTrue(np.allclose(normalized_arr, expected_result),
                         'Arrays are not equal')

    def test_get_normalized_vector_zero_array_as_input(self):
        '''Test normalized vectors of zeros array as input'''
        inp_arr = np.array([0, 0, 0])
        normalized_arr = MathTools.get_normalized_vector(inp_arr)

        self.assertTrue(np.array_equal(normalized_arr, inp_arr),
                         'Arrays are not equal')

    def test_make_orthogonal_succesfully_done(self):
        '''Test ortogonal of vactor1 onto vector2 with valid input'''
        vector1 = np.array([1, 1, 3])
        vector2 = np.array([1, 1, 0])
        result = MathTools.make_orthogonal(vector1, vector2)
        expected_result = np.array([-1, -1,  3])
        
        self.assertTrue(np.array_equal(expected_result, result))    


















if __name__ == '__main__':
	unittest.main()

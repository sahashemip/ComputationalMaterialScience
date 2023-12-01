import unittest
import yaml
import tempfile
import numpy as np
from pathlib import Path
from polarization_orientation_Raman import InputParser

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

		# Delete the file to simulate file not found
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
											[7, 7, 7]
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
										[1, 1, 1]
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
											[8, 8, 3]
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
	
if __name__ == '__main__':
	unittest.main()

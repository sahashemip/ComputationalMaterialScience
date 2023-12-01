'''
	The code compute polarization-orientation Raman intensity.
	
	Classes:
		- RamanScattering
		-
		-
	
	Methods:
		-
		-
		-
		
	Example Usage:
	
	
	Author: Arsalan Hahsemi
			sahashemip@gmail.com
'''

import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from pathlib import Path

class InputParser:
	'''
	Parses input file give in path to input_file.
	
	Methods:
		-
		-
		-
		-
		
	Attributes:
		-
		-
		-
	
	Example Usage:
		
	'''
	
	def __init__(self, input_file: Path) -> None:
		'''
		Intializes class RamanScattering with path of input_file.
		'''
		#TODO Path or str
		if not isinstance(input_file, Path):
			raise TypeError('Expected "input_file" to be a Path object')

		if not input_file.exists():
			raise FileNotFoundError(f'File {input_file} does not exist!')
		
		self.input_file = input_file
		
	def read_input(self) -> Optional[dict]:
		'''
        Reads the input file and returns its content as a dictionary.

        Returns:
            dict|None : The content of the YAML file.

        Raises:
            ValueError: If the file content is not a valid YAML format.
            OSError: If the file is not found.
		'''
		try:
			with open(self.input_file, 'r') as inpfile:
				return yaml.safe_load(inpfile)

		except yaml.YAMLError as e:
			raise yaml.scanner.ScannerError(f'Error in {self.input_file}!', e)
		
		except FileNotFoundError as e:
			raise OSError(f'File {self.input_file} not found!')

	def parse_raman_tensors(self) -> any:
		'''
		Returns 3x3 Raman tensors in a list.
		Raises a KeyError if the 'ramantensor'key not found in the
		 input yaml file.
		'''
		inp_data = self.read_input()
		raman_tensors = []
		raman_tensor_key = 'ramantensor'
		
		try:
			raman_tensors_ = inp_data[raman_tensor_key]
			
			for key in raman_tensors_.keys():
				raman_tensor_of_each_key = raman_tensors_[key]
				
				if not np.array(raman_tensor_of_each_key).shape == (3, 3):
					raise ValueError('Matrix shape is not 3x3!')
				raman_tensors.append(raman_tensor_of_each_key)
			
			return raman_tensors
			
		except KeyError:
			raise KeyError(f"Key '{raman_tensor_key}' not found in input!")

			


parsereader = InputParser(Path('input.yaml'))
print(parsereader.parse_raman_tensors())
#print(parsereader.read_input())
'''
class RamanScattering:
	"""
	
	"""

	



		
	def parse_propagation_vectors(self) -> list:
		"""
		
		"""
		propagation_vectors = []
		propagation_vector_data = self.read_input()['propagationvector']
		for key in propagation_vector_data.keys():
			propagation_vectors.append(propagation_vector_data[key])
		return propagation_vectors

	def parse_polarization_vectors(self) -> list:
		"""
		
		"""
		polarization_vectors = []
		polarization_vector_data = self.read_input()['polarizationvector']
		for key in polarization_vector_data.keys():
			polarization_vectors.append(polarization_vector_data[key])
		return polarization_vectors

	@staticmethod
	def normalize_vector(vector: np.ndarray) -> np.ndarray:
		return vector / np.linalg.norm(vector)

	@staticmethod
	def make_polarization_orthogonal(polarization: np.ndarray, propagation: np.ndarray):
		return polarization - np.dot(polarization, propagation) * propagation

	@staticmethod
	def form_coordinate_system(basis_1: np.ndarray, basis_2: np.ndarray):
		basis_3 = np.cross(basis_1, basis_2)
		return np.array([basis_1, basis_3, basis_2])

	@staticmethod
	def get_beam_polarization(basis_set: np.ndarray, material_coord: np.ndarray, axis: np.ndarray):
		new_basis_inv = np.linalg.inv(basis_set)
		return np.dot(np.dot(np.dot(new_basis_inv, material_coord), basis_set), axis.T)

	#@staticmethod
	def compute_parallel_raman(self, basis_set: np.ndarray, axis: np.ndarray, max_angle: int):
		print(max_angle)
		intensity = np.zeros(max_angle)
		print(intensity)
		alpha_range = np.arange(max_angle) / 180 * np.pi
		for i, alpha in enumerate(alpha_range):
			material_coord = np.array(
			[[np.cos(alpha), -np.sin(alpha), 0],
			[np.sin(alpha), np.cos(alpha), 0],
			[0, 0, 1]]
			)
			
			ei = RamanScattering.get_beam_polarization(basis_set, material_coord, axis)
			
			for R in self.parse_raman_tensors():
				intensity[i] += np.abs(np.dot(np.dot(ei.T, np.array(R)), ei))**2
		return np.column_stack((alpha_range, intensity))

	def compute_perpendicualr_raman(basis_set: np.ndarray, axis1: np.ndarray, axis2: np.ndarray, max_angle: int=360):
		intensity = np.zeros(max_angle)
		alpha_range = np.arange(max_angle) / 180 * np.pi
		for i, alpha in enumerate(alpha_range):
			material_coord = np.array(
			[[np.cos(alpha), -np.sin(alpha), 0],
			[np.sin(alpha), np.cos(alpha), 0],
			[0, 0, 1]]
			)
			
			ei = get_beam_polarization(basis_set, material_coord, axis1)
			es = get_beam_polarization(basis_set, material_coord, axis2)
			
			for R in self.parse_raman_tensors():
				intensity[i] += np.abs(np.dot(np.dot(es.T, np.array(R)), ei))**2
		return np.column_stack((alpha_range, intensity))
	
	@staticmethod
	def make_polarplot(alpha, intensity, output_fig: Path='polarplot,png'):
		normal_intensity = intensity / np.max(intensity)
		plt.polar(alpha, normal_intensity, 'r-')
		plt.savefig(output_fig, dpi=600)

	@staticmethod
	def write_to_file(combined_array: np.ndarray, output_file: Path='output.txt'):
		np.savetxt(output_file, combined_array, fmt='%d', delimiter='\t')

#class RamanCalculator:
#	def __init__(self, raman):
#		self.raman = raman
	
raman = RamanScattering('input.yaml')
raman_tensors = np.array(raman.parse_raman_tensors())
propagations = np.array(raman.parse_propagation_vectors())
polarization = np.array(raman.parse_polarization_vectors())

for i, q_vector in enumerate(propagations):
	q_vector = raman.normalize_vector(q_vector) #q
	#print(q_vector)
	polarization = raman.make_polarization_orthogonal(polarization, q_vector)
	polarization = raman.normalize_vector(polarization[0]) #a
	#print(polarization)
	basis_sets = raman.form_coordinate_system(polarization, q_vector)
	#print(basis_sets)
	raman_data = raman.compute_parallel_raman(basis_sets, polarization, 360)

'''

	
	
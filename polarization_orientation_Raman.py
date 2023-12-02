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

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml

class InputParser:
    '''
	Parses input file give in path to input_file.
	
	Methods:
		- read_input
		- parse_raman_tensors
		-
		-
		
	Attributes:
		- input_file: Path to input_file
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
        Returns a list of 3x3 Raman tensors.
        Raises a KeyError if the 'ramantensor'key not found in
        the input yaml file.
        '''
        raman_tensors = []
        raman_tensor_key = 'ramantensor'
        
        try:
            raman_tensors_ = self.read_input()[raman_tensor_key]
            
            for key in raman_tensors_.keys():
                raman_tensor_of_each_key = raman_tensors_[key]
                
                if not np.array(raman_tensor_of_each_key).shape == (3, 3):
                    raise ValueError('Matrix shape is not 3x3!')
                raman_tensors.append(np.array(raman_tensor_of_each_key))

            return raman_tensors

        except KeyError:
            raise KeyError(f"Key '{raman_tensor_key}' not found in input!")

    def parse_propagation_vectors(self) -> any:
        """
        Returns a list of  3x1 matrices.
        Raises a KeyError if the 'propagationvector'key not found in
        the input yaml file.
        """
        propagation_vectors = []
        propagat_vector_key = 'propagationvector'
        
        try:
            propagat_vectors_ = self.read_input()[propagat_vector_key]
            
            for key in propagat_vectors_.keys():
                propagat_vector_of_each_key = propagat_vectors_[key]
                
                if not np.array(propagat_vector_of_each_key).shape == (3,):
                    raise ValueError('Matrix shape is not 3x1!')
                propagation_vectors.append(np.array(propagat_vector_of_each_key))

            return propagation_vectors

        except KeyError:
            raise KeyError(f"Key '{propagat_vector_key}' not found in input!")

    def parse_polarization_vectors(self) -> any:
        """
        Returns a 3x1 matrix in list.
        Raises a KeyError if the 'polarizationvector'key not found in
        the input yaml file.
        """
        polarization_vectors = []
        polarization_vector_key = 'polarizationvector'
        
        try:
            polarization_vectors_ = self.read_input()[polarization_vector_key]
            
            for key in polarization_vectors_.keys():
                polarization_vectors_each_key = polarization_vectors_[key]
                
                if not np.array(polarization_vectors_each_key).shape == (3,):
                    raise ValueError('Matrix shape is not 3x1!')
                polarization_vectors.append(np.array(polarization_vectors_each_key))

            return polarization_vectors

        except KeyError:
            raise KeyError(f"Key '{polarization_vector_key}' not found in input!")


class MathTools:
    '''
    Performs several linear algebra.
    
	Methods:
		- get_normalized_vector
		- 
		-
		-
	
	Example Usage:
	  
    '''
    @staticmethod
    def get_normalized_vector(input_arr: np.ndarray) -> np.ndarray:
        '''Normalizes the input NumPy array'''
        norm_of_array = np.linalg.norm(input_arr)
        if norm_of_array == 0:
            return input_arr
        return input_arr / norm_of_array

    @staticmethod
    def make_orthogonal(vector1: np.ndarray, vector2: np.ndarray):
        '''Calculates the projection of vector1 onto vector2'''
        return vector1 - np.dot(vector1, vector2) * vector2
    
    #Should we normalize each vector first?
    @staticmethod
    def create_coordinate_system(vector1: np.ndarray, vector2: np.ndarray):
        '''Returns all 3 axes of a coordinate system as a 3x3 array.'''
        vector1 = MathTools.get_normalized_vector(vector1)
        vector2 = MathTools.get_normalized_vector(vector2)

        vector3 = np.cross(vector1, vector2)
        vector3 = MathTools.get_normalized_vector(vector3)
        
        if np.all(vector3 == 0):
            raise ValueError('Coordinate system in not generated by parallel bases.')
        return np.array([vector1, vector3, vector2])

    @staticmethod
    def rotate_material_coordinare_around_z(alpha: float) -> np.ndarray:
        ''' Return a rotated coordinate system around z-axis'''
        return np.array([[np.cos(alpha), -np.sin(alpha), 0],
                         [np.sin(alpha), np.cos(alpha), 0], 
                         [0, 0, 1]],)

    #TODO: what does this vector does?    
    @staticmethod
    def get_beam_polarization_vector(basis_set: np.ndarray,
                                     material_coord: np.ndarray, axis: np.ndarray):
        '''Returns polarization vector as a 3x1 array'''
        
        new_basis_inv = np.linalg.inv(basis_set)
        return np.dot(np.dot(np.dot(new_basis_inv, material_coord), basis_set), axis.T)


class RamanCalculator:
    '''
    Computes polarization-orientation Raman intensity
    
    
    Methods:
        - 
        -
        -
    
    Attributes:
        - inputparser: an instace of class InputParser
        - mathtools: an instance of class MathTools
    '''
    def __init__(self, inputparser, mathtools):
        self.inputparser = inputparser
        self.mathtools = mathtools
        

    def compute_raman(self):
        raman_tensors = self.inputparser.parse_raman_tensors()
        #print(raman_tensors)
        propagation_vectors = self.inputparser.parse_propagation_vectors()
        #print(propagation_vectors)
        polarization_vectors = self.inputparser.parse_polarization_vectors()
        #print(polarization_vectors)

        for polvect in polarization_vectors:
            polvect = self.mathtools.get_normalized_vector(polvect)
            for propvect in propagation_vectors:
                propvect = self.mathtools.get_normalized_vector(propvect)
                
                a = self.mathtools.make_orthogonal(polvect, propvect)
                nb = self.mathtools.create_coordinate_system(polvect, propvect)
                b = nb[1]
                raman_intensities = np.zeros(360)
                alphas = np.arange(360)

                for i, alpha in enumerate(alphas):
                    alpha = alpha / 180 * np.pi
                    m = self.mathtools.rotate_material_coordinare_around_z(alpha)
                    
                    ei = self.mathtools.get_beam_polarization_vector(nb, m, a)

                    for R in raman_tensors:
                        raman_intensities[i] += (np.abs(np.dot(ei.T, (np.dot(R, ei)))))**2
                
                plt.polar(alphas, raman_intensities, 'r-')
                plt.show()





import matplotlib.pyplot as plt

parsereader = InputParser(Path('input.yaml'))
mathtools = MathTools.get_normalized_vector(np.array([3, 4, 5]))

raman = RamanCalculator(InputParser(Path('input.yaml')), MathTools())
raman.compute_raman()
#print(parsereader.parse_polarization_vectors())
#print(parsereader.read_input())
'''
class RamanScattering:

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

	
	

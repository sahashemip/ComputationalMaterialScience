'''
    The code compute polarization-orientation Raman intensity.

    Classes:
        - InputParser
        - MathTools
        - RamanCalculator

    Example Usage:
        - python polarization_orientation_Raman.py -i input.yaml -p

    Author: Arsalan Hahsemi
            sahashemip@gmail.com
'''

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


class InputParser:
    '''
        Parses input file give in path to input_file.

        Methods:
            - read_input
            - parse_raman_tensors
            - parse_propagation_vectors
            - parse_polarization_vectors

    Attributes:
            - input_file: Path to input_file
    '''

    def __init__(self, input_file: Path) -> None:
        '''
        Initializes class RamanScattering with path of input_file.
        '''
        if not isinstance(input_file, Path):
            raise TypeError('Expected "input_file" to be a Path object')

        if not input_file.exists():
            raise FileNotFoundError(f'File {input_file} does not exist!')

        self.input_file = input_file

    def read_input(self) -> any:
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

        except yaml.YAMLError:
            raise yaml.scanner.ScannerError(f'Error in {self.input_file}!')

        except FileNotFoundError:
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
                propagation_vectors.append(
                    np.array(propagat_vector_of_each_key)
                )

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
                polarization_vectors.append(
                    np.array(polarization_vectors_each_key)
                )

            return polarization_vectors

        except KeyError:
            raise KeyError(
                f"Key '{polarization_vector_key}' not found in input!"
            )


class MathTools:
    '''
        Performs several linear algebra.

        Methods:
            - get_normalized_vector
            - make_orthogonal
            - create_coordinate_system
            - rotate_material_coordinare_around_z
            - get_beam_polarization_vector
            - compute_raman
            - make_polarplot
            - write_to_file
    '''
    @staticmethod
    def get_normalized_vector(input_array: np.ndarray) -> np.ndarray:
        '''Normalizes the input NumPy array'''
        norm_of_array = np.linalg.norm(input_array)
        if norm_of_array == 0:
            return input_array
        return input_array / norm_of_array

    @staticmethod
    def make_orthogonal(vector1: np.ndarray, vector2: np.ndarray):
        '''Calculates the projection of vector1 onto vector2'''
        return vector1 - np.dot(vector1, vector2) * vector2

    @staticmethod
    def create_coordinate_system(vector1: np.ndarray,
                                 vector2: np.ndarray):
        '''Returns all 3 axes of a coordinate system as a 3x3 array.'''
        vector3 = np.cross(vector1, vector2)

        if np.all(vector3 == 0):
            raise ValueError(
                'Coordinate system in not generated by parallel bases.'
            )
        return np.array([vector2, vector3, vector1])

    @staticmethod
    def rotate_material_coordinare_around_z(alphas: np.ndarray) -> np.ndarray:
        ''' Return a rotated coordinate system around z-axis
            alphas (np.ndarray): An array of angles in radians.
        '''
        cos_alphas, sin_alphas = np.cos(alphas), np.sin(alphas)
        rotation_matrices = np.zeros((len(alphas), 3, 3))

        rotation_matrices[:, 0, 0] = cos_alphas
        rotation_matrices[:, 0, 1] = -sin_alphas
        rotation_matrices[:, 1, 0] = sin_alphas
        rotation_matrices[:, 1, 1] = cos_alphas
        rotation_matrices[:, 2, 2] = 1

        return rotation_matrices

    @staticmethod
    def get_beam_polarization_vector(basis_set: np.ndarray,
                                     material_coord: np.ndarray,
                                     axis: np.ndarray):
        '''Returns polarization vector as a 3x1 array'''

        return np.linalg.inv(basis_set) @ material_coord @ basis_set @ axis.T

    @staticmethod
    def compute_raman(vector1: np.ndarray,
                      tensor: np.ndarray,
                      vector2: np.ndarray) -> float:
        '''Returns Raman activity (a float number) for incident
           beam polarization vector (vector1), Raman tensor
           (3x3 matrix) and scattered beam vector (vector2).'''

        return np.abs(vector2.T @ tensor @ vector1)**2

    @staticmethod
    def make_polarplot(angles,
                       intensities,
                       output_fig: Path = 'polarplot.png',
                       color='red',
                       label=None):
        '''Returns a file of polar plot.'''
        plt.polar(angles, intensities, color=color, label=label)
        plt.legend()
        plt.savefig(output_fig, dpi=600)

    @staticmethod
    def write_to_file(combined_array: np.ndarray,
                      output_file: Path = 'output.txt'):
        '''Returns a text file o angles and intensities.'''
        np.savetxt(output_file, combined_array, fmt='%d', delimiter='\t')


class RamanCalculator:
    '''
        Computes polarization-orientation Raman intensity

        Methods:
            - compute_polarization_orientation_raman
            -

        Attributes:
            - inputparser: an instace of class InputParser
            - mathtools: an instance of class MathTools
    '''

    number_of_grids = 360

    def __init__(self, inputparser, mathtools, is_plotting=True):
        self.inputparser = inputparser
        self.mathtools = mathtools
        self.is_plotting = is_plotting
        
    def compute_polarization_orientation_raman(self):
        raman_tensors = self.inputparser.parse_raman_tensors()
        propagation_vectors = self.inputparser.parse_propagation_vectors()
        polarization_vectors = self.inputparser.parse_polarization_vectors()

        alphas = np.arange(RamanCalculator.number_of_grids)
        alphas_in_radian = np.radians(alphas)

        m = self.mathtools.rotate_material_coordinare_around_z(
                alphas_in_radian
                )

        for i, polarization_vector in enumerate(polarization_vectors):
            for j, propagation_vector in enumerate(propagation_vectors):
                
                norm_propvect = self.mathtools.get_normalized_vector(propagation_vector)
                
                axis1 = self.mathtools.make_orthogonal(
                                polarization_vector,
                                norm_propvect
                                )
                norm_axis1 = self.mathtools.get_normalized_vector(axis1)
                axis2 = np.cross(norm_propvect, norm_axis1)

                coordinate_system = self.mathtools.create_coordinate_system(
                                    norm_propvect,
                                    norm_axis1
                                    )
                
                intensity_vv = np.zeros(RamanCalculator.number_of_grids)
                intensity_hv = np.zeros(RamanCalculator.number_of_grids)

                for k, alpha in enumerate(alphas):
                    ei = self.mathtools.get_beam_polarization_vector(
                                coordinate_system, m[k], norm_axis1
                                )
                    es = self.mathtools.get_beam_polarization_vector(
                                coordinate_system, m[k], axis2
                                )

                    for R in raman_tensors:
                        intensity_vv[k] += self.mathtools.compute_raman(ei, R, ei)
                        intensity_hv[k] += self.mathtools.compute_raman(ei, R, es)
                        
                if self.is_plotting:
                    for intensity, labl, col in [(intensity_vv, 'VV', 'blue'),
                                                 (intensity_hv, 'HV', 'red')]:
                        self.mathtools.make_polarplot(angles=alphas_in_radian,
                                                      intensities=intensity,
                                                      output_fig=f'fig-{i}-{j}.png',
                                                      color=col,
                                                      label=labl)
                    plt.figure().clear()

parser = argparse.ArgumentParser(description="Run Raman Calculator")

parser.add_argument('-i', '--input', type=str, required=True, help="Path to input.yaml file")
parser.add_argument('-p', '--plot', action='store_true', help="Enable plotting")

# Step 3: Parse arguments
args = parser.parse_args()

input_path = Path(args.input)
is_plotting = args.plot

raman = RamanCalculator(InputParser(input_path), MathTools(), is_plotting=is_plotting)
raman.compute_polarization_orientation_raman()



	
	

from pathlib import Path

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
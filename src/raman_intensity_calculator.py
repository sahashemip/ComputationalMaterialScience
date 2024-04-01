'''
    The code compute polarization-orientation Raman intensity.

    Class:
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

import utils as utils
from input_parser import InputParser as inputparser


class RamanCalculator:
    '''
        Computes polarization-orientation Raman intensity

        Methods:
            - compute_polarization_orientation_raman
            -

        Attributes:
            - inputparser: an instace of class InputParser
            - utils: an instance of class utils
    '''

    def __init__(self,
                 inputparser,
                 upper_limit_degree: int = 360,
                 spectrocopy_configuration: str = 'both',
                 ):
        self.inputparser = inputparser
        self.number_of_grids = upper_limit_degree
        self.configuration = spectrocopy_configuration

    def __get_parallel_configuration_intensity(
        self,
        coordinate_system: np.ndarray,
        material_coordinate: np.ndarray,
        axis
        ) -> np.ndarray:

        intensity = np.zeros(self.number_of_grids)
        
        for grid in range(self.number_of_grids):
            ei = utils.get_beam_polarization_vector(
                coordinate_system,
                material_coordinate[grid],
                axis
                )
            
            for R in self.inputparser.parse_raman_tensors():
                intensity[grid] += utils.compute_raman(ei, R, ei)
        return intensity

    def __get_cross_configuration_intensity(
        self,
        coordinate_system: np.ndarray,
        material_coordinate: np.ndarray,
        axis1: np.ndarray,
        axis2: np.ndarray,
        ) -> np.ndarray:

        intensity = np.zeros(self.number_of_grids)
        
        for grid in range(self.number_of_grids):
            ei = utils.get_beam_polarization_vector(
                coordinate_system,
                material_coordinate[grid],
                axis1
                )
            
            es = utils.get_beam_polarization_vector(
                coordinate_system,
                material_coordinate[grid],
                axis2
                )
            for R in self.inputparser.parse_raman_tensors():
                intensity[grid] += utils.compute_raman(ei, R, es)
        return intensity

    def compute_polarization_orientation_raman(self):
        propagation_vectors = self.inputparser.parse_propagation_vectors()
        polarization_vectors = self.inputparser.parse_polarization_vectors()

        angles_in_radian = utils.get_degrees_in_radian(self.number_of_grids)
        material_coordinate = utils.rotate_material_coordinate_around_z(
            angles_in_radian)

        for i, polarization_vector in enumerate(polarization_vectors):
            for j, propagation_vector in enumerate(propagation_vectors):
                
                norm_propvect = utils.get_normalized_vector(propagation_vector)
                
                axis1 = utils.make_orthogonal(polarization_vector, norm_propvect)
                norm_axis1 = utils.get_normalized_vector(axis1)
        
                axis2 = np.cross(norm_propvect, norm_axis1)

                coordinate_system = utils.create_coordinate_system(norm_propvect,
                                                                   norm_axis1)
                
                if self.configuration == 'parallel' or self.configuration == 'both':
                    intensity_vv = self.__get_parallel_configuration_intensity(
                        coordinate_system,
                        material_coordinate,
                        norm_axis1,
                        )
                    utils.write_to_npzfile(
                        angles_in_radian,
                        intensity_vv,
                        outfile=f'polvect-{i}-propvect-{j}-parallel-configuration.npz')
                
                if self.configuration == 'cross' or self.configuration == 'both':
                    intensity_hv = self.__get_cross_configuration_intensity(
                        coordinate_system,
                        material_coordinate,
                        norm_axis1,
                        axis2,
                        )
                    utils.write_to_npzfile(
                        angles_in_radian,
                        intensity_hv,
                        outfile=f'polvect-{i}-propvect-{j}-cross-configuration.npz')
	

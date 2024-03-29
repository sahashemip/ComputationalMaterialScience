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

import utils as mathtools
from input_parser import InputParser as inputparser


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

    def __init__(self, inputparser, is_plotting=True):
        self.inputparser = inputparser
        self.is_plotting = is_plotting
        
    def compute_polarization_orientation_raman(self):
        raman_tensors = self.inputparser.parse_raman_tensors()
        propagation_vectors = self.inputparser.parse_propagation_vectors()
        polarization_vectors = self.inputparser.parse_polarization_vectors()

        alphas = np.arange(RamanCalculator.number_of_grids)
        alphas_in_radian = np.radians(alphas)

        m = mathtools.rotate_material_coordinare_around_z(
                alphas_in_radian
                )

        for i, polarization_vector in enumerate(polarization_vectors):
            for j, propagation_vector in enumerate(propagation_vectors):
                
                norm_propvect = mathtools.get_normalized_vector(propagation_vector)
                
                axis1 = mathtools.make_orthogonal(
                                polarization_vector,
                                norm_propvect
                                )
                norm_axis1 = mathtools.get_normalized_vector(axis1)
                axis2 = np.cross(norm_propvect, norm_axis1)

                coordinate_system = mathtools.create_coordinate_system(
                                    norm_propvect,
                                    norm_axis1
                                    )
                
                intensity_vv = np.zeros(RamanCalculator.number_of_grids)
                intensity_hv = np.zeros(RamanCalculator.number_of_grids)

                for k, alpha in enumerate(alphas):
                    ei = mathtools.get_beam_polarization_vector(
                                coordinate_system, m[k], norm_axis1
                                )
                    es = mathtools.get_beam_polarization_vector(
                                coordinate_system, m[k], axis2
                                )

                    for R in raman_tensors:
                        intensity_vv[k] += mathtools.compute_raman(ei, R, ei)
                        intensity_hv[k] += mathtools.compute_raman(ei, R, es)
                        
                if self.is_plotting:
                    for intensity, labl, col in [(intensity_vv, 'VV', 'blue'),
                                                 (intensity_hv, 'HV', 'red')]:
                        mathtools.make_polarplot(angles=alphas_in_radian,
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

raman = RamanCalculator(inputparser(input_path), is_plotting=is_plotting)
raman.compute_polarization_orientation_raman()



	
	

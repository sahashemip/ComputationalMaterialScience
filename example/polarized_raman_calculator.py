'''
    This code computes polarization propagation Raman intensity.

    Author: Arsalan Hahsemi
            sahashemip@gmail.com
'''

import argparse
from pathlib import Path
from intensity_calculator import RamanCalculator
from input_parser import InputParser

def main():
    parser = argparse.ArgumentParser(description="Run Raman Intensity Calculator ...")

    parser.add_argument('-i',
                        '--input',
                        type=str,
                        required=True,
                        help="Path to the input yamlfile")
    
    parser.add_argument('-conf',
                        '--configuration',
                        type=str,
                        help="Set up experimental configuration: parallel, cross, both")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    configuration = args.configuration
    
    raman = RamanCalculator(
        InputParser(input_path),
        spectrocopy_configuration=configuration)
    raman.compute_polarization_orientation_raman()

if __name__ == "__main__":
    main()

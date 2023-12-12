
from pathlib import Path
from ase.io import read
import yaml


class RamanScattering:
	def __init__(self, poscar_file: Path, output_file: Path):
		self.poscar_file = poscar_file
		self.output_file = output_file

class PoscarParser:
	def __init__(self, poscar_file: Path):
		self.poscar_file = poscar_file

	def read_poscar(self):
		poscar = read(self.poscar_file)
		return poscar

	def get_masses(self):
		return self.parse_poscar().get_masses()

	def get_atomic_coordinates(self):
		return self.parse_poscar().get_positions()

	def get_cell_vectors(self):
		return self.parse_poscar().get_cells()


class QpointsYamlReader:
	def __init__(self, phonopy_yaml_file: Path):
		self.phonopy_yaml_file = phonopy_yaml_file

	def read_yaml_file(self):
	
	def get_eigenvalues(self):
	
	def get_eigenvectors(self):


class RamanInputMaker:
	"""
	
	"""
	def make_folders(self):
		
	def make_inputs(self):
	
	def

class PolarizabilityCalculator:
	"""
	
	"""
	def get_dielectric_constants(self):
	
	def get_dispacements(self)
	
	def get_polarizability_tensor(self):


class OutputWriter(RamanScattering):
	def write_to_output(self):
		with open(self.output_file)
	
	








	

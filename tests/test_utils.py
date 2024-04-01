import os
import unittest
from pathlib import Path
import numpy as np

from utils import (
    get_normalized_vector,
    make_orthogonal,
    create_coordinate_system,
    rotate_material_coordinate_around_z,
    get_beam_polarization_vector,
    compute_raman,
    write_to_npzfile
    )


class TestLinearAlgebraFunctions(unittest.TestCase):
    """Unittests for methods in '../src/utils.py' module."""

    def test_get_normalized_vector_with_non_zero_input(self):
        """Test normalization of a non-zero vector."""
        input_array = np.array([3, 4])
        expected_output = np.array([0.6, 0.8])
        np.testing.assert_array_almost_equal(get_normalized_vector(input_array),
                                             expected_output)
    
    def test_get_normalized_vector_with_zero_input(self):
        """Test normalization of a zero vector."""
        input_array = np.array([0, 0])
        np.testing.assert_array_almost_equal(get_normalized_vector(input_array),
                               input_array)

    def test_make_orthogonal(self):
        """Test orthogonal projection is computed correctly."""
        vector1 = np.array([3, 4])
        vector2 = np.array([4, 3])
        expected_projection = vector1 - np.dot(vector1, vector2) * vector2
        np.testing.assert_array_almost_equal(make_orthogonal(vector1, vector2), expected_projection)

    def test_create_coordinate_system_normal_case(self):
        """Test that the function returns a valid coordinate system for non-parallel vectors."""
        vector1 = np.array([1, 0, 0])
        vector2 = np.array([0, 1, 0])
        expected_vector3 = np.cross(vector1, vector2)
        expected_output = np.array([vector2, expected_vector3, vector1])
        
        result = create_coordinate_system(vector1, vector2)
        np.testing.assert_array_equal(result, expected_output)

    def test_create_coordinate_system_error_for_parallel_vectors(self):
        """Test that the function raises a ValueError for parallel vectors."""
        vector1 = np.array([1, 1, 1])
        vector2 = np.array([2, 2, 2])
        
        with self.assertRaises(ValueError) as context:
            create_coordinate_system(vector1, vector2)  
        self.assertTrue('Coordinate system is not generated by parallel bases.' in str(context.exception))

    def test_create_coordinate_system_error_for_zero_vector(self):
        """Test that the function raises a ValueError when one of the vectors is a zero vector."""
        vector1 = np.array([0, 0, 0])
        vector2 = np.array([1, 0, 0])

        with self.assertRaises(ValueError) as context:
            create_coordinate_system(vector1, vector2)        
        self.assertTrue('Coordinate system is not generated by parallel bases.' in str(context.exception))

    def test_rotate_material_coordinate_around_z_single_angle(self):
        """Test rotation with a single angle."""
        alpha = np.array([np.pi / 2])
        expected_matrix = np.array([
            [[0, -1, 0],
             [1, 0, 0],
             [0, 0, 1]]
        ])
        result = rotate_material_coordinate_around_z(alpha)
        np.testing.assert_array_almost_equal(result, expected_matrix)

    def test_rotate_material_coordinate_around_z_multiple_angles(self):
        """Test rotation with multiple angles."""
        alphas = np.array([0, np.pi / 2, np.pi, 2 * np.pi])
        expected_matrices = np.array([
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]],
            [[0, -1, 0],
             [1, 0, 0],
             [0, 0, 1]],
            [[-1, 0, 0],
             [0, -1, 0],
             [0, 0, 1]],
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
        ])
        result = rotate_material_coordinate_around_z(alphas)
        np.testing.assert_array_almost_equal(result, expected_matrices)

    def test_rotate_material_coordinate_around_z_edge_cases(self):
        """Test rotation with edge case angles."""
        alphas = np.array([-2 * np.pi, 4 * np.pi])
        expected_matrices = np.array([
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]],
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
        ])
        result = rotate_material_coordinate_around_z(alphas)
        np.testing.assert_array_almost_equal(result, expected_matrices)

    def test_get_beam_polarization_vector(self):
        """Test that the function returns the correct polarization vector."""
        basis_set = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        material_coord = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        axis = np.array([1, 0, 0])
        expected_result = np.linalg.inv(basis_set) @ material_coord @ basis_set @ axis.T
        result = get_beam_polarization_vector(basis_set, material_coord, axis)
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_compute_raman_basic(self):
        """Test computing Raman activity with basic inputs."""
        vector1 = np.array([1, 0, 0])
        vector2 = np.array([0, 1, 0])
        tensor = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        expected = 0
        
        result = compute_raman(vector1, tensor, vector2)
        self.assertEqual(result, expected)

    def test_compute_raman_with_non_identity_tensor(self):
        """Test with a non-identity tensor to verify non-zero Raman activity."""
        
        vector1 = np.array([1, 0, 0])
        vector2 = np.array([0, 1, 0])
        non_identity_tensor = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        expected_non_zero = np.abs(vector2.T @ non_identity_tensor @ vector1)**2
        result = compute_raman(vector1, non_identity_tensor, vector2)
        self.assertEqual(result, expected_non_zero)

    def test_compute_raman_edge_case(self):
        """Test computing Raman activity with edge-case values."""
        zero_vector = np.array([0, 0, 0])
        vector2 = np.array([0, 1, 0])
        tensor = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        result = compute_raman(zero_vector, tensor, vector2)
        self.assertEqual(result, 0)

    def setUp(self):
        # Temporary file path for testing
        self.test_file = Path("test_data.npz")
        self.x = np.array([1, 2, 3])
        self.y = np.array([4, 5, 6])

    def tearDown(self):
        # Cleanup: Remove the file after testing
        if self.test_file.exists():
            os.remove(self.test_file)

    def test_write_to_npzfile_file_creation(self):
        """Test if the file is created."""
        write_to_npzfile(self.x, self.y, self.test_file)
        self.assertTrue(self.test_file.exists(), "The NPZ file should exist after writing.")

    def test_write_to_npzfile_data_integrity(self):
        """Test if the data saved in the file matches the input."""
        write_to_npzfile(self.x, self.y, self.test_file)
        with np.load(self.test_file) as data:
            np.testing.assert_array_equal(data['x'], self.x, "The 'x' data should match the input.")
            np.testing.assert_array_equal(data['y'], self.y, "The 'y' data should match the input.")


if __name__ == '__main__':
    print('Testing utils.py module')
    unittest.main()

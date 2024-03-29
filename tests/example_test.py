# arithmetic.py

class Arithmetic:
    """Class for performing basic arithmetic operations."""

    @staticmethod
    def add(x, y):
        """Return the sum of x and y."""
        return x + y

    @staticmethod
    def subtract(x, y):
        """Return the difference of x and y."""
        return x - y




# test_arithmetic.py

import unittest
from arithmetic import Arithmetic

class TestArithmetic(unittest.TestCase):
    """Test cases for Arithmetic operations."""

    def test_add(self):
        """Test the add method."""
        self.assertEqual(Arithmetic.add(10, 5), 15)
        self.assertEqual(Arithmetic.add(-1, 1), 0)
        self.assertEqual(Arithmetic.add(-1, -1), -2)

    def test_subtract(self):
        """Test the subtract method."""
        self.assertEqual(Arithmetic.subtract(10, 5), 5)
        self.assertEqual(Arithmetic.subtract(-1, 1), -2)
        self.assertEqual(Arithmetic.subtract(-1, -1), 0)

if __name__ == '__main__':
    unittest.main()


python test_arithmetic.py

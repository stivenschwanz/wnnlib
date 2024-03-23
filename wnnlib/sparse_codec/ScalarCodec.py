import unittest
from numpy import random
import numpy as np
from wnnlib.BitUtils import BitUtils


class ScalarCodec:
    """
    A simply scalar encoder/decoder (codec).

    """

    def __init__(self, min_value, max_value, total_number_of_bits, number_of_active_bits, obfuscate=False):
        """
        Initialize the sparse codec.

        Parameters:
            min_value (float): Minimum allowed value to encode.
            max_value (float): Maximum allowed value to encode.
            total_number_of_bits (int): Sparse binary representation length.
            number_of_active_bits (int): Number of active bits in the sparse representation.
            obfuscate (bool): Use obfuscate lib.
        """

        # ----------------------------------------------------------------------
        # Check parameters
        # ----------------------------------------------------------------------

        # Check the minimum against the maximum allowed values
        assert min_value < max_value

        # Check the total number of bits against the number of activated bits
        assert total_number_of_bits > number_of_active_bits > 0

        # ----------------------------------------------------------------------
        # Save the parameters
        # ----------------------------------------------------------------------
        self.min_value = min_value
        self.max_value = max_value
        self.total_number_of_bits = total_number_of_bits
        self.number_of_active_bits = number_of_active_bits

        # ----------------------------------------------------------------------
        # Initialize the encoder
        # ----------------------------------------------------------------------

        # Compute the number of buckets
        self.number_of_buckets = self.total_number_of_bits - self.number_of_active_bits + 1

        # Compute the range
        self.range = self.max_value - self.min_value

    def encode(self, input_value):
        """
        Encode a scalar value into a high-dimensional, sparse binary representation.

        Parameters:
            input_value (float): an input value in the observation space.

        Returns:
            (bool[]): High-dimensional vector containing a sparse binary representation of the given dense vector.
        """

        # Check the input value against the minimum and maximum values
        assert self.min_value <= input_value <= self.max_value

        # Determine the initial bucket index
        initial_active_bit_index = int(np.floor(self.number_of_buckets*(input_value-self.min_value)/self.range))
        final_active_bit_index = initial_active_bit_index + self.number_of_active_bits

        # Get the sparse vector according to the selected index
        sparse_vector = np.zeros(self.total_number_of_bits, order='C', dtype=bool)
        sparse_vector[initial_active_bit_index:final_active_bit_index] = True

        return sparse_vector

    def decode(self, sparse_vector):
        """
        Decode a sparse vector into a scalar value.

        Parameters:
            sparse_vector (bool[]): Sparse binary representation.

        Returns:
            (float): A decoded value in the observation space.
        """

        if sparse_vector is None:
            return None

        active_bit_indexes = np.asarray(np.nonzero(sparse_vector))[0]

        mean_active_bit_index = np.mean(active_bit_indexes)
        initial_active_bit_index = int(np.clip(mean_active_bit_index - self.number_of_active_bits/2, 0, self.total_number_of_bits))
        decoded_value = initial_active_bit_index * float(self.range) / float(self.number_of_buckets) + self.min_value

        return decoded_value


class TestScalarCodec(unittest.TestCase):
    """
    Extends unittest.TestCase class to implement unit tests for the ScalarCodec class.
    """

    def test_0_codec(self):
        """
        Test case 1: load generated sparse vectors.
        """
        scalar_codec = ScalarCodec(min_value=0, max_value=100, total_number_of_bits=2048, number_of_active_bits=64)

        input_values = 100*np.random.random(size=10)
        for input_value in input_values:
            print('input_value=', input_value)
            sparse_vector = scalar_codec.encode(input_value)
            print('sparce_vector=', sparse_vector)
            decoded_value = scalar_codec.decode(sparse_vector)
            print('decoded_value=', decoded_value)


if __name__ == '__main__':
    unittest.main()

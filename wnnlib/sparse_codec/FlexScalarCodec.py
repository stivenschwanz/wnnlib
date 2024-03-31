import unittest
from numpy import random
import numpy as np
import math
from wnnlib.BitUtils import BitUtils
from wnnlib.sparse_codec import ScalarCodec


class FlexScalarCodec:
    """
    A flexible scalar encoder/decoder (codec).

    """

    def __init__(self, total_number_of_bits, number_of_active_bits, obfuscate=False):
        """
        Initialize the sparse codec.

        Parameters:
            total_number_of_bits (int): Sparse binary representation length.
            number_of_active_bits (int): Number of active bits in the sparse representation.
            obfuscate (bool): Use obfuscate lib.
        """

        # ----------------------------------------------------------------------
        # Check parameters
        # ----------------------------------------------------------------------

        # Check the total number of bits against the number of activated bits
        assert total_number_of_bits > number_of_active_bits > 0

        # ----------------------------------------------------------------------
        # Save the parameters
        # ----------------------------------------------------------------------
        self.min_value = -2 ** 63
        self.max_value = +2 ** 64
        self.total_number_of_bits = total_number_of_bits
        self.number_of_active_bits = number_of_active_bits

        # ----------------------------------------------------------------------
        # Initialize the encoder
        # ----------------------------------------------------------------------

        # Initialize the mantissa scalar codec
        self.mantissa_total_number_of_bits = math.floor(total_number_of_bits / 2)
        self.mantissa_number_of_active_bits = math.floor(number_of_active_bits / 2)
        self.mantissa_codec = ScalarCodec.ScalarCodec(-1.0, 1.0, self.mantissa_total_number_of_bits,
                                                      self.mantissa_number_of_active_bits)

        # Initialize the exponent scalar codec
        self.exponent_total_number_of_bits = total_number_of_bits - self.mantissa_total_number_of_bits
        self.exponent_number_of_active_bits = number_of_active_bits - self.mantissa_number_of_active_bits
        self.exponent_codec = ScalarCodec.ScalarCodec(-63, 64, self.exponent_total_number_of_bits,
                                                      self.exponent_number_of_active_bits)

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

        # Get mantissa + exponent
        (m, e) = np.frexp(input_value)

        # Encode the mantissa + exponent
        sparse_vector = np.concatenate((self.mantissa_codec.encode(m), self.exponent_codec.encode(e)))

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

        # Decode the mantissa + exponent
        m = self.mantissa_codec.decode(sparse_vector[:self.mantissa_total_number_of_bits])
        e = self.exponent_codec.decode(sparse_vector[self.mantissa_total_number_of_bits:])
        i = int(np.round(e))

        # Combine the mantissa + exponent
        decoded_value = np.ldexp(m, i)

        return decoded_value


class TestFlexScalarCodec(unittest.TestCase):
    """
    Extends unittest.TestCase class to implement unit tests for the FlexScalarCodec class.
    """

    def test_0_codec(self):
        """
        Test case 1: load generated sparse vectors.
        """
        scalar_codec = FlexScalarCodec(total_number_of_bits=2048, number_of_active_bits=64)

        input_values = 10000*np.random.random(size=10)
        for input_value in input_values:
            print('------------------------------')
            print('input_value=', input_value)
            sparse_vector = scalar_codec.encode(input_value)
            print('sparce_vector=', sparse_vector)
            decoded_value = scalar_codec.decode(sparse_vector)
            print('decoded_value=', decoded_value)


if __name__ == '__main__':
    unittest.main()

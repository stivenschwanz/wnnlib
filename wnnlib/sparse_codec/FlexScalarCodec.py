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
        self.min_value = -2 ** 95
        self.max_value = +2 ** 95
        self.total_number_of_bits = total_number_of_bits
        self.number_of_active_bits = number_of_active_bits

        # ----------------------------------------------------------------------
        # Initialize the encoder
        # ----------------------------------------------------------------------

        # Initialize the mantissa scalar codec
        self.mantissa1_total_number_of_bits = 815
        self.mantissa1_number_of_active_bits = 16
        self.mantissa1_number_of_skip_bits = 4
        self.mantissa1_codec = ScalarCodec.ScalarCodec(-100, 100, self.mantissa1_total_number_of_bits,
                                                       self.mantissa1_number_of_active_bits,
                                                       self.mantissa1_number_of_skip_bits)

        # Initialize the mantissa scalar codec
        self.mantissa2_total_number_of_bits = 103
        self.mantissa2_number_of_active_bits = 4
        self.mantissa2_number_of_skip_bits = 1
        self.mantissa2_codec = ScalarCodec.ScalarCodec(0, 99, self.mantissa2_total_number_of_bits,
                                                       self.mantissa2_number_of_active_bits,
                                                       self.mantissa2_number_of_skip_bits)
        # Initialize the exponent scalar codec
        self.exponent_total_number_of_bits = 821
        self.exponent_number_of_active_bits = 32
        self.exponent_number_of_skip_bits = 8
        self.exponent_codec = ScalarCodec.ScalarCodec(-48, 49, self.exponent_total_number_of_bits,
                                                      self.exponent_number_of_active_bits,
                                                      self.exponent_number_of_skip_bits)

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

        # Breakdown the mantissa
        q = 0
        s = 1
        r = []
        for i in range(0, 5):
            s = 100*s
            p = int(np.floor(s * m))
            r.append(p - 100*q)
            q = p

        # Encode the mantissa + exponent
        sparse_vector = np.concatenate((self.mantissa1_codec.encode(r[0]),
                                        self.mantissa2_codec.encode(r[1]),
                                        self.mantissa2_codec.encode(r[2]),
                                        self.mantissa2_codec.encode(r[3]),
                                        self.mantissa2_codec.encode(r[4]),
                                        self.exponent_codec.encode(e)))

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
        start_bit = 0
        end_bit = self.mantissa1_total_number_of_bits
        r = int(self.mantissa1_codec.decode(sparse_vector[start_bit:end_bit]))
        s = 0.01
        m = s * r
        for i in range(0, 4):
            start_bit = end_bit
            end_bit += self.mantissa2_total_number_of_bits
            r = int(self.mantissa2_codec.decode(sparse_vector[start_bit:end_bit]))
            s *= 0.01
            m += s * r
        start_bit = end_bit
        e = self.exponent_codec.decode(sparse_vector[start_bit:])
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
        Test case 0: load generated sparse vectors.
        """
        scalar_codec = FlexScalarCodec(total_number_of_bits=1024, number_of_active_bits=40)

        input_values = 1000000*np.random.random(size=10)
        for input_value in input_values:
            print('------------------------------')
            print('input_value=', input_value)
            sparse_vector = scalar_codec.encode(input_value)
            print('sparce_vector=', sparse_vector)
            decoded_value = scalar_codec.decode(sparse_vector)
            print('decoded_value=', decoded_value)

    def test_1_codec(self):
        """
        Test case 1: load generated sparse vectors.
        """
        scalar_codec = FlexScalarCodec(total_number_of_bits=1024, number_of_active_bits=40)

        for delta in [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0, 1000.0, 10000.0]:
            input_value1 = 10000.0
            input_value2 = input_value1 - delta

            print('------------------------------')
            print('input_value1=', input_value1)
            sparse_vector1 = scalar_codec.encode(input_value1)
            print('sparce_vector1=', sparse_vector1)
            decoded_value1 = scalar_codec.decode(sparse_vector1)
            print('decoded_value1=', decoded_value1)

            print('------------------------------')
            print('input_value2=', input_value2)
            sparse_vector2 = scalar_codec.encode(input_value2)
            print('sparce_vector2=', sparse_vector2)
            decoded_value2 = scalar_codec.decode(sparse_vector2)
            print('decoded_value2=', decoded_value2)

            print('------------------------------')
            print('delta=', delta)
            print('input_value1-input_value2=', input_value1 - input_value2)
            print('d_H(sparce_vector1,sparce_vector2)=', np.count_nonzero(sparse_vector1 != sparse_vector2))
            print('decoded_value1-decoded_value2=', decoded_value1 - decoded_value2)


if __name__ == '__main__':
    unittest.main()

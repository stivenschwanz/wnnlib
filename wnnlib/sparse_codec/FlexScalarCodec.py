import unittest
from numpy import random
import numpy as np
import math
from wnnlib.BitUtils import BitUtils
from wnnlib.sparse_codec import ScalarCodec


def frexp10(x):
    e = int(math.ceil(math.log10(abs(x)))) if x != 0 else 0
    m = x / 10 ** e
    return m, e


def ldexp10(m, e):
    return m * (10 ** e)


class FlexScalarCodec:
    """
    A flexible scalar encoder/decoder (codec).

    """

    def __init__(self, min_exponent=-48, max_exponent=+49,
                 mantissa_number_of_active_bits=[8, 4, 2],
                 mantissa_number_of_skip_bits=[3, 2, 1],
                 exponent_number_of_active_bits=16,
                 exponent_number_of_skip_bits=4,
                 # min_exponent=-48, max_exponent=+49,
                 # mantissa_number_of_active_bits=[16, 8, 4, 2],
                 # mantissa_number_of_skip_bits=[4, 3, 2, 1],
                 # exponent_number_of_active_bits=32,
                 # exponent_number_of_skip_bits=9,
                 obfuscate=False):
        """
        Initialize the sparse codec.

        Parameters:
            min_exponent (int): Minimum exponent of an encodable input value.
            max_exponent (int): Maximum exponent of an encodable input value.
            mantissa_number_of_active_bits (int[]): Number of active bits for the mantissa1 sparse representation.
            mantissa_number_of_skip_bits (int[]): Number of skip bits (Hamming distance for delta 1) for the mantissa1 sparse representation.
            exponent_number_of_active_bits (int): Number of active bits for the exponent sparse representation.
            exponent_number_of_skip_bits (int): Number of skip bits (Hamming distance for delta 1) for the exponent sparse representation.
            obfuscate (bool): Use obfuscate lib.
        """

        # ----------------------------------------------------------------------
        # Save the parameters
        # ----------------------------------------------------------------------
        self.min_value = -10 ** max_exponent
        self.max_value = +10 ** max_exponent
        self.number_of_mantissa_codecs = len(mantissa_number_of_active_bits)
        self.mantissa_number_of_active_bits = mantissa_number_of_active_bits
        self.mantissa_number_of_skip_bits = mantissa_number_of_skip_bits
        self.exponent_number_of_active_bits = exponent_number_of_active_bits
        self.exponent_number_of_skip_bits = exponent_number_of_skip_bits

        # ----------------------------------------------------------------------
        # Check parameters
        # ----------------------------------------------------------------------

        # Check the number of parameters are consistent
        assert self.number_of_mantissa_codecs > 0
        assert self.number_of_mantissa_codecs == len(self.mantissa_number_of_skip_bits)

        # Check the number of activated bits against the number of skip bits
        for i in range(0, self.number_of_mantissa_codecs):
            assert self.mantissa_number_of_active_bits[i] >= self.mantissa_number_of_skip_bits[i] > 0
        assert exponent_number_of_active_bits > exponent_number_of_skip_bits > 0

        # ----------------------------------------------------------------------
        # Initialize the encoder
        # ----------------------------------------------------------------------

        # Initialize the total number of useful bits
        self.total_number_of_useful_bits = 0

        # Initialize the mantissa scalar codecs
        self.mantissa_min_value = -50
        self.mantissa_max_value = 50
        self.mantissa_number_of_buckets = self.mantissa_max_value - self.mantissa_min_value + 1
        self.mantissa_codecs = []
        self.mantissa_total_number_of_bits = []
        for i in range(0, self.number_of_mantissa_codecs):
            self.mantissa_total_number_of_bits.append(self.mantissa_number_of_buckets * \
                                                      self.mantissa_number_of_skip_bits[i] + \
                                                      self.mantissa_number_of_active_bits[i] - 1)
            self.total_number_of_useful_bits += self.mantissa_total_number_of_bits[i]
            self.mantissa_codecs.append(ScalarCodec.ScalarCodec(self.mantissa_min_value, self.mantissa_max_value,
                                                                self.mantissa_total_number_of_bits[i],
                                                                self.mantissa_number_of_active_bits[i],
                                                                self.mantissa_number_of_skip_bits[i]))

        # Initialize the exponent scalar codec
        self.exponent_min_value = min_exponent
        self.exponent_max_value = max_exponent
        self.exponent_number_of_buckets = self.exponent_max_value - self.exponent_min_value + 1

        exponent_total_number_of_bits = self.exponent_number_of_buckets * self.exponent_number_of_skip_bits + \
                                        self.exponent_number_of_active_bits - 1
        self.total_number_of_useful_bits += exponent_total_number_of_bits
        self.exponent_codec = ScalarCodec.ScalarCodec(self.exponent_min_value, self.exponent_max_value,
                                                      exponent_total_number_of_bits,
                                                      self.exponent_number_of_active_bits,
                                                      self.exponent_number_of_skip_bits)

        # Trailling bits
        self.total_number_of_bits = 2**int(math.ceil(math.log2(self.total_number_of_useful_bits)))
        self.number_of_trailling_bits = self.total_number_of_bits - self.total_number_of_useful_bits
        self.trailling_bits = np.zeros(self.number_of_trailling_bits, order='C', dtype=bool)
        self.last_useful_bit = self.total_number_of_bits - self.number_of_trailling_bits

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

        # Get mantissa + exponent (base 10)
        (m, e) = frexp10(input_value)

        # The magic happens here
        m /= 2

        # Breakdown the mantissa
        q = 0
        s = 1
        r = []
        for i in range(0, self.number_of_mantissa_codecs):
            s = 100 * s
            p = int(np.round(s * m))
            r.append(p - 100 * q)
            q = p

        # Encode the mantissa + exponent
        sparse_vector_list = []
        for i in range(0, self.number_of_mantissa_codecs):
            sparse_vector_list.append(self.mantissa_codecs[i].encode(r[i]))
        sparse_vector_list.append(self.exponent_codec.encode(e))
        sparse_vector_list.append(self.trailling_bits)
        sparse_vector = np.concatenate(sparse_vector_list)

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
        end_bit = 0
        s = 1
        m = 0
        for i in range(0, self.number_of_mantissa_codecs):
            start_bit = end_bit
            end_bit += self.mantissa_total_number_of_bits[i]
            r = int(np.round(self.mantissa_codecs[i].decode(sparse_vector[start_bit:end_bit])))
            s *= 0.01
            m += s * r
        start_bit = end_bit
        end_bit = self.last_useful_bit
        e = self.exponent_codec.decode(sparse_vector[start_bit:end_bit])
        i = int(np.round(e))

        # Undoing the magic
        m *= 2

        # Combine the mantissa + exponent (base 10)
        decoded_value = ldexp10(m, i)

        return decoded_value


class TestFlexScalarCodec(unittest.TestCase):
    """
    Extends unittest.TestCase class to implement unit tests for the FlexScalarCodec class.
    """

    def test_0_codec(self):
        """
        Test case 0: load generated sparse vectors.
        """
        scalar_codec = FlexScalarCodec(min_exponent=-48, max_exponent=+49,
                                       mantissa_number_of_active_bits=[8, 4, 2],
                                       mantissa_number_of_skip_bits=[3, 2, 1],
                                       exponent_number_of_active_bits=16,
                                       exponent_number_of_skip_bits=4)

        input_values = 1000000 * np.random.random(size=10)
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
        scalar_codec = FlexScalarCodec()

        for delta in [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0, 1000.0,
                      10000.0]:
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

    def test_2_codec(self):
        """
        Test case 2: trying out small deltas to check for big changes
        """
        scalar_codec = FlexScalarCodec()

        delta = -1.0
        input_value1 = 160000.0
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

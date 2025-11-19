import unittest
from numpy import random
import numpy as np
from wnnlib.utils.BitUtils import BitUtils


class ScalarCodec:
    """
    A simply scalar encoder/decoder (codec).

    """

    def __init__(self, min_value, max_value, res_value,
                 number_of_active_bits=16,
                 number_of_skip_bits=1,
                 number_of_gap_bits=0):
        """
        Initialize the sparse codec.

        Parameters:
            min_value (float): Minimum allowed value to encode.
            max_value (float): Maximum allowed value to encode.
            res_value (float): Resolution to encode.
            number_of_active_bits (int): Number of active bits in the sparse representation.
            number_of_skip_bits (int): Number of bits to skip in the sparse representation of consecutive values.
            number_of_gap_bits (int): Number of gap bits.
        """

        # ----------------------------------------------------------------------
        # Check parameters
        # ----------------------------------------------------------------------

        # Check the minimum against the maximum allowed values
        assert min_value < max_value

        # Check the resolution against the range of allowed values
        assert max_value - min_value > res_value

        # Check the total number of bits against the number of activated bits
        assert number_of_active_bits > 0

        # Check the number of bits against the number of activated bits
        assert number_of_active_bits >= number_of_skip_bits > 0

        # Check the number of bits against the number of gap bits
        assert number_of_gap_bits >= 0

        # ----------------------------------------------------------------------
        # Save the parameters
        # ----------------------------------------------------------------------
        self.min_value = min_value
        self.max_value = max_value
        self.res_value = res_value
        self.number_of_active_bits = number_of_active_bits
        self.number_of_skip_bits = number_of_skip_bits
        self.number_of_gap_bits = number_of_gap_bits

        # ----------------------------------------------------------------------
        # Initialize the encoder
        # ----------------------------------------------------------------------

        # Compute the range
        self.range_value = self.max_value - self.min_value

        # Compute the number of buckets
        self.number_of_buckets = int(np.ceil(self.range_value / self.res_value)) + 1

        # Compute the total number of bits
        if self.number_of_gap_bits > 0:
            self.total_number_of_bits = int(np.ceil(self.number_of_buckets * self.number_of_skip_bits / \
                           (self.number_of_active_bits - 1)) + self.number_of_active_bits + self.number_of_gap_bits)
        else:
            self.total_number_of_bits = self.number_of_buckets * self.number_of_skip_bits + self.number_of_active_bits

        # Precomputing encoding constants
        self.e1 = +np.float64(self.total_number_of_bits - self.number_of_active_bits - self.number_of_gap_bits) / np.float64(self.range_value)
        self.e2 = -self.e1 * np.float64(self.min_value)

        if self.number_of_gap_bits > 0:
            self.e3 = (self.number_of_active_bits - 1) * self.e1
            self.e4 = 0
        else:
            self.e3 = 0
            self.e4 = 0

        # Precomputing decoding constants
        self.d1 = +1.0 / self.e1
        self.d2 = -self.e2 / self.e1

    def get_total_number_of_bits(self):
        """
        Get the total umber of bits required to properly encoded the input values with the desired resolution.

        Returns:
           (int): The total number of bits.
        """
        return self.total_number_of_bits

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
        n_init = self.e1 * input_value + self.e2
        n_skip = int(np.floor(2*n_init)/2)

        input_skip = self.d1 * n_skip + self.d2
        input_delta = input_value - input_skip

        # Determine the gap bucket index
        n_gap = np.clip(int(np.floor(self.e3 * input_delta + self.e4)), 0, self.number_of_active_bits - 1)

        # Get the sparse vector according to the selected indexes
        seq = [(n_skip, BitUtils.low),
               (self.number_of_active_bits - n_gap, BitUtils.high),
               (self.number_of_gap_bits, BitUtils.low),
               (n_gap, BitUtils.high),
               (self.total_number_of_bits - self.number_of_active_bits - self.number_of_gap_bits - n_skip, BitUtils.low)]

        return BitUtils.sparse_vector(seq)

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

        # Determine the average index
        n_avg = BitUtils.mean_index(sparse_vector)

        # Determine the initial bucket index
        n_dec = n_avg - (self.number_of_active_bits - 1) / 2

        # Decode the initial bucket index
        decoded_value = self.d1 * n_dec + self.d2

        return decoded_value


class TestScalarCodec(unittest.TestCase):
    """
    Extends unittest.TestCase class to implement unit tests for the ScalarCodec class.
    """

    def test_0_codec(self):
        """
        Test case 0: check if decoded values are compatible with the configured resolution.
        """

        resolution = 0.5

        self.scalar_codec = ScalarCodec(min_value=-1000,
                                        max_value=1000,
                                        res_value=resolution,
                                        number_of_active_bits=16,
                                        number_of_skip_bits=2,
                                        number_of_gap_bits=1)

        input_values = np.random.random_integers(low=-1000, high=1000, size=1000)
        for input_value in input_values:
            print('input_value=', input_value)
            sparse_vector = self.scalar_codec.encode(input_value)
            # print('sparce_vector=', sparse_vector)
            decoded_value = self.scalar_codec.decode(sparse_vector)
            print('decoded_value=', decoded_value)
            print('decoded_value-input_value=', decoded_value-input_value)
            assert np.abs(decoded_value - input_value) < resolution
            print('(decoded_value-input_value)/input_value (%)=', 100*(decoded_value-input_value)/input_value)

    def test_1_codec(self):
        """
        Test case 1:  check if decoded values are compatible with the configured resolution.
        """
        resolution = 0.001

        self.scalar_codec = ScalarCodec(min_value=-1,
                                        max_value=1,
                                        res_value=resolution,
                                        number_of_active_bits=32,
                                        number_of_skip_bits=2,
                                        number_of_gap_bits=1)

        input_values = 2*np.random.random(size=1000) - 1
        for input_value in input_values:
            print('input_value=', input_value)
            sparse_vector = self.scalar_codec.encode(input_value)
            print('sparce_vector=', sparse_vector)
            decoded_value = self.scalar_codec.decode(sparse_vector)
            print('decoded_value=', decoded_value)
            print('decoded_value-input_value=', decoded_value-input_value)
            assert np.abs(decoded_value-input_value) < resolution
            print('(decoded_value-input_value)/input_value (%)=', 100*(decoded_value-input_value)/input_value)

    def test_2_codec(self):
        """
        Test case 2:  check if decoded values are compatible with the configured resolution.
        """
        resolution = 1

        self.scalar_codec = ScalarCodec(min_value=-50,
                                        max_value=50,
                                        res_value=resolution,
                                        number_of_active_bits=16,
                                        number_of_skip_bits=6,
                                        number_of_gap_bits=0)

        #input_values = 2 * np.random.random(size=1000) - 1
        #for input_value in input_values:
        for input_value in [1, 2]:
            print('input_value=', input_value)
            sparse_vector = self.scalar_codec.encode(input_value)
            print('sparce_vector=', sparse_vector)
            decoded_value = self.scalar_codec.decode(sparse_vector)
            print('decoded_value=', decoded_value)
            print('decoded_value-input_value=', decoded_value - input_value)
            assert np.abs(decoded_value - input_value) < resolution
            print('(decoded_value-input_value)/input_value (%)=', 100 * (decoded_value - input_value) / input_value)


if __name__ == '__main__':
    unittest.main()

import unittest
from numpy import random
import numpy as np
import math
from wnnlib.BitUtils import BitUtils
from collections import deque

MIN_RANGE = 1e-4


class AdaptiveScalarCodec:
    """
    An adaptive scalar encoder/decoder (codec) using a sliding window to determine the minimum and maximum values for
    the scalar codec.

    """
    def __init__(self, number_of_active_bits=32,
                 total_number_of_bits=1024,
                 max_window_size=100,
                 min_value=None,
                 max_value=None):
        """
        Initialize the sparse codec.

        Parameters:

        """

        # ----------------------------------------------------------------------
        # Check parameters
        # ----------------------------------------------------------------------

        # Check the total number of bits against the number of activated bits
        assert total_number_of_bits > number_of_active_bits > 0

        # Check the window size
        assert max_window_size >= 2

        # ----------------------------------------------------------------------
        # Store input parameters
        # ----------------------------------------------------------------------
        self.number_of_active_bits = number_of_active_bits
        self.total_number_of_bits = total_number_of_bits
        self.max_window_size = max_window_size

        # ----------------------------------------------------------------------
        # Initialize the class members
        # ----------------------------------------------------------------------

        # Compute the number of buckets
        self.number_of_buckets = self.total_number_of_bits - self.number_of_active_bits + 1

        # Invalidate codec bounds
        self.min_value = min_value
        self.max_value = max_value
        if min_value is not None and max_value is not None:
            self.range_value = max_value - min_value
        else:
            self.range_value = None

        # Invalidate codec constants
        self.e1 = None
        self.e2 = None
        self.d1 = None
        self.d2 = None

        # ----------------------------------------------------------------------
        # Initialize an empty sliding window to store the last input samples
        # ----------------------------------------------------------------------
        self.sliding_window = deque([], self.max_window_size)

        # ----------------------------------------------------------------------
        # Initialize an empty dictionary to store known bounds
        # ----------------------------------------------------------------------
        self.bounds_map = {}

    def update_bounds(self, input_value):
        # Sanity check
        if self.min_value is not None and \
                self.max_value is not None and \
                self.min_value <= input_value <= self.max_value:
            # NOP: the input value is inside the current codec bounds
            return

        # Update sliding window
        self.sliding_window.appendleft(input_value)
        if len(self.sliding_window) > self.max_window_size:
            self.sliding_window.pop()

        # Update bounds
        self.min_value = min(self.sliding_window)
        self.max_value = max(self.sliding_window)

        # Sanity check
        if self.max_value - self.min_value < MIN_RANGE:
            self.max_value = self.min_value + MIN_RANGE

    def update_codec_constants(self):
        # Update the codec range
        self.range_value = self.max_value - self.min_value

        # Compute encoding constants
        self.e1 = +np.float64(self.total_number_of_bits - self.number_of_active_bits) / np.float64(self.range_value)
        self.e2 = -self.e1 * np.float64(self.min_value)

        # Compute decoding constants
        self.d1 = +1.0 / self.e1
        self.d2 = -self.e2 / self.e1

    def adapt_codec(self, input_value):
        # Update bounds first
        self.update_bounds(input_value)

        # Then, update codec constants
        self.update_codec_constants()

    def encode(self, input_value):
        """
        Encode a scalar value into a high-dimensional, sparse binary representation.

        Parameters:
            input_value (float): an input value in the observation space.

        Returns:
            (bool[]): High-dimensional vector containing a sparse binary representation of the given dense vector.
        """

        # Update codec
        self.adapt_codec(input_value)

        # Determine the initial bucket index
        n_init = self.e1 * input_value + self.e2
        n_skip = int(np.floor(2*n_init)/2)

        # Get the sparse vector according to the selected indexes
        seq = [(n_skip, BitUtils.low),
               (self.number_of_active_bits, BitUtils.high),
               (self.total_number_of_bits - self.number_of_active_bits - n_skip, BitUtils.low)]

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


class TestAdaptiveScalarCodec(unittest.TestCase):
    """
    Extends unittest.TestCase class to implement unit tests for the AdaptiveScalarCodec class.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up method:
        """
        cls.scalar_codec = AdaptiveScalarCodec(number_of_active_bits=48,
                                               total_number_of_bits=2048,
                                               max_window_size=10)

    @classmethod
    def tearDownClass(cls):
        """
        Tear down method:
        """
        cls.scalar_codec = None

    def test_0_codec(self):
        """
        Test case 0:
        """

        input_values1 = 10 * np.random.random(size=10)
        for input_value in input_values1:
            print('------------------------------')
            print('input_value=', input_value)
            sparse_vector = self.scalar_codec.encode(input_value)
            print('sparce_vector=', sparse_vector)
            decoded_value = self.scalar_codec.decode(sparse_vector)
            print('decoded_value=', decoded_value)

        input_values2 = 100 * np.random.random(size=10)
        for input_value in input_values2:
            print('------------------------------')
            print('input_value=', input_value)
            sparse_vector = self.scalar_codec.encode(input_value)
            print('sparce_vector=', sparse_vector)
            decoded_value = self.scalar_codec.decode(sparse_vector)
            print('decoded_value=', decoded_value)

        for input_value in input_values1:
            print('------------------------------')
            print('input_value=', input_value)
            sparse_vector = self.scalar_codec.encode(input_value)
            print('sparce_vector=', sparse_vector)
            decoded_value = self.scalar_codec.decode(sparse_vector)
            print('decoded_value=', decoded_value)

if __name__ == '__main__':
    unittest.main()

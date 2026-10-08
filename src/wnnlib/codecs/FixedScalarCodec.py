import numpy as np
from wnnlib.utils.BitUtils import BitUtils

MIN_RANGE = 1e-2


class FixedScalarCodec:
    """
    An adaptive scalar encoder/decoder (codec) using a sliding window to determine the minimum and maximum values for
    the scalar codec.

    """
    def __init__(self, min_value, max_value,
                 number_of_active_bits=32,
                 total_number_of_bits=1024):
        """
        Initialize the sparse codec.

        Parameters:

        """

        # ----------------------------------------------------------------------
        # Check parameters
        # ----------------------------------------------------------------------

        # Check the minimum against the maximum allowed values
        assert min_value <= max_value

        # Check the total number of bits against the number of activated bits
        assert total_number_of_bits > number_of_active_bits > 0

        # ----------------------------------------------------------------------
        # Store input parameters
        # ----------------------------------------------------------------------
        self.min_value = min_value
        self.max_value = max_value
        self.number_of_active_bits = number_of_active_bits
        self.total_number_of_bits = total_number_of_bits

        # ----------------------------------------------------------------------
        # Initialize the class members
        # ----------------------------------------------------------------------

        # Compute the number of buckets
        self.number_of_buckets = self.total_number_of_bits - self.number_of_active_bits + 1

        # Sanity check
        if self.max_value - self.min_value < MIN_RANGE:
            self.max_value = self.min_value + MIN_RANGE

        # Compute the codec range
        self.range_value = self.max_value - self.min_value

        # Compute encoding constants
        self.e1 = +np.float64(self.total_number_of_bits - self.number_of_active_bits) / np.float64(self.range_value)
        self.e2 = -self.e1 * np.float64(self.min_value)

        # Compute decoding constants
        self.d1 = +1.0 / self.e1
        self.d2 = -self.e2 / self.e1

    def encode(self, input_value):
        """
        Encode a scalar value into a high-dimensional, sparse binary representation.

        Parameters:
            input_value (float): an input value in the observation space.

        Returns:
            (bool[]): High-dimensional vector containing a sparse binary representation of the given dense vector.
        """

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

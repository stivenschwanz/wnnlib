import unittest
import numpy as np
from wnnlib.codecs.FlexScalarCodec import FlexScalarCodec


class TestFlexScalarCodec(unittest.TestCase):
    """
    Extends unittest.TestCase class to implement unit tests for the FlexScalarCodec class.
    """

    def test_0_codec(self):
        """
        Test case 0: load generated sparse vectors.
        """
        scalar_codec = FlexScalarCodec(min_exponent=-59, max_exponent=+59,
                                       mantissa_number_of_active_bits=[32],
                                       mantissa_number_of_skip_bits=[2],
                                       mantissa_number_of_gap_bits=[1],
                                       exponent_number_of_active_bits=16,
                                       exponent_number_of_skip_bits=6,
                                       exponent_number_of_gap_bits=0)

        input_values = 1000000 * np.random.random(size=10)
        for input_value in input_values:
            print('------------------------------')
            print('input_value=', input_value)
            sparse_vector = scalar_codec.encode(input_value)
            #print('sparce_vector=', sparse_vector)
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

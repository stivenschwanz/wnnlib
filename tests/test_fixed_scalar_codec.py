import unittest
import numpy as np
from wnnlib.codecs.FixedScalarCodec import FixedScalarCodec


class TestFixedScalarCodec(unittest.TestCase):
    """
    Extends unittest.TestCase class to implement unit tests for the FixedScalarCodec class.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up method:
        """
        cls.scalar_codec = FixedScalarCodec(0, 100, number_of_active_bits=48, total_number_of_bits=2048)

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

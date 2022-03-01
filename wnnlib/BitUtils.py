import unittest
import numpy as np
import time


class BitUtils:

    @staticmethod
    def hamming_distance(a, b):
        # return np.sum(np.bitwise_xor(a, b))
        return np.count_nonzero(a != b)

    @staticmethod
    def binary_array_to_integer(arr):
        val = np.int64(0)
        pow2 = np.int64(1)
        length = len(arr)
        for i in range(length-1, -1, -1):
            val += pow2 * arr[i]
            pow2 *= np.int64(2)
        return val

    @staticmethod
    def integer_to_binary_array(val, length):
        arr = np.zeros(length, order='C', dtype=np.uint8)
        i = length-1
        while val > 0 and i >= 0:
            arr[i] = val & 1
            val >>= 1
            i -= 1
        return arr

    @staticmethod
    def integer_to_binary_array2(val, length):
        arr = np.zeros(length, order='C', dtype=np.uint8)
        #arr = []
        i = 0
        for bit in bin(val)[2:].zfill(length):
        #for bit in format(val, "016b"):
            arr[i] = np.uint8(bit)
            #arr.append(np.uint8(bit))
            i += 1
        return arr


class TestBitUtils(unittest.TestCase):
    """
    Extends unittest.TestCase class to implement unit tests for the VGRAMArray class.
    """

    """
        Extends unittest.TestCase class to implement unit tests for the VGRAMArray class.
        """
    number_of_patterns = 256
    pattern_length = 16
    test_statistics = None

    @classmethod
    def setUpClass(cls):
        """
        Set up method: configure parameters and create a VGRAM node.
        """
        cls.test_statistics = {"average_binary_array_to_integer_time": 0.0,
                               "average_integer_to_binary_array_time": 0.0,
                               "elapsed_binary_array_to_integer_time": 0.0,
                               "elapsed_integer_to_binary_array_time": 0.0}

    @classmethod
    def tearDownClass(cls):
        """
        Tear down method: print test statistics.
        """
        print('Elapsed time to convert {0} patterns: {1:.2e} seconds'.format(cls.number_of_patterns,
                                                                             cls.test_statistics["elapsed_binary_array_to_integer_time"]))
        print('Average convert time: {:.2e} seconds'.format(cls.test_statistics["average_binary_array_to_integer_time"]))
        print('Elapsed time to convert {0} patterns: {1:.2e} seconds'.format(cls.number_of_patterns,
                                                                             cls.test_statistics["elapsed_binary_array_to_integer_time"]))
        print('Average convert time: {:.2e} seconds'.format(cls.test_statistics["elapsed_integer_to_binary_array_time"]))

        cls.test_statistics = None

    def test_0_learn(self):
        """
        Test case 0: binary array to integer.
        """
        elapsed_time1 = 0
        elapsed_time2 = 0
        for n in range(0, self.number_of_patterns):
            arr1 = np.random.randint(low=0, high=2, size=self.pattern_length, dtype=np.uint8)
            t = time.time()
            val1 = BitUtils.binary_array_to_integer(arr1)
            elapsed_time1 += time.time() - t
            t = time.time()
            arr2 = BitUtils.integer_to_binary_array(val1, self.pattern_length)
            elapsed_time2 += time.time() - t
            val2 = BitUtils.binary_array_to_integer(arr2)
            assert val1 == val2

        self.test_statistics["elapsed_binary_array_to_integer_time"] = elapsed_time1
        self.test_statistics["average_binary_array_to_integer_time"] = elapsed_time1 / self.number_of_patterns
        self.test_statistics["elapsed_integer_to_binary_array_time"] = elapsed_time2
        self.test_statistics["average_integer_to_binary_array_time"] = elapsed_time2 / self.number_of_patterns


if __name__ == '__main__':
    unittest.main()

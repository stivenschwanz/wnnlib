import unittest
import numpy as np
import time
import sys
from wnnlib.codecs.KDTree import KDTree


def exec_codec_test(tree: KDTree, data, statistics, style, dx, dy):
    """
    Execute a test.

    Parameters:
        tree (object): kd-tree.
        data (double[]): Data points.
        statistics (dict[]) : Test statistics
        style (int): Tree style (0: rectangles, 1: sectors)
    """
    np.random.seed(0)
    number_of_points = data.size/2
    elapsed_time1 = 0
    elapsed_time2 = 0
    acc_error2 = 0
    for dense_vector1 in data:
        # Encoding
        t = time.time()
        sparse_vector = tree.encode(dense_vector=dense_vector1)
        elapsed_time1 += time.time() - t

        # Debug kd-tree
        if not "pytest" in sys.modules:
            tree.debug(style=style, dx=dx, dy=dy)

        # Decoding
        t = time.time()
        dense_vector2 = tree.decode(sparse_vector=sparse_vector)
        elapsed_time2 += time.time() - t

        # Accumulated squared error
        squared_error = np.transpose(dense_vector1 - dense_vector2) * (dense_vector1 - dense_vector2)
        acc_error2 += squared_error

        print('-----------------------------------------')
        print('Dense vector 1 = %s ' % dense_vector1)
        print('Dense vector 1 = %s ' % dense_vector2)
        print('Squared error = %s ' % squared_error)

    statistics["number_of_encoding_points"] = number_of_points
    statistics["elapsed_encoding_time"] = elapsed_time1
    statistics["average_encoding_time"] = elapsed_time1 / number_of_points
    statistics["number_of_decoding_points"] = number_of_points
    statistics["elapsed_decoding_time"] = elapsed_time2
    statistics["average_decoding_time"] = elapsed_time2 / number_of_points
    statistics["rms_decoding_error"] = np.sqrt(acc_error2 / number_of_points)


class TestKDTree(unittest.TestCase):
    """
    Extends unittest.TestCase class to implement unit tests for the KDTree class.
    """

    test_0_tree = None
    test_0_data = None
    test_0_statistics = None
    test_1_tree = None
    test_1_data = None
    test_1_statistics = None
    test_2_tree = None
    test_2_data = None
    test_2_statistics = None

    @classmethod
    def setUpClass(cls):
        """
        Set up method: configure parameters and create kd-trees.
        """
        np.random.seed(0)
        cls.test_0_tree = KDTree(max_depth=16, learning_rate=0.001, min_splitting_volume=0.00001,
                                 min_bounds=[0, 0], max_bounds=[10, 10],
                                 depth=0, sparse_vectors_file="./data/64k_sparse_vectors_seed_0.npz")
        cls.test_0_data = np.append(np.random.uniform(low=0, high=10, size=[64, 2]),
                                    np.random.multivariate_normal(mean=[5, 5], cov=[[1, 0.5], [0.5, 1]], size=64),
                                    axis=0)
        cls.test_0_statistics = {"number_of_encoding_points": 0.0,
                                 "elapsed_encoding_time": 0.0,
                                 "average_encoding_time": 0.0,
                                 "number_of_decoding_points": 0.0,
                                 "elapsed_decoding_time": 0.0,
                                 "average_decoding_time": 0.0,
                                 "rms_decoding_error": [0.0, 0.0]}
        cls.test_1_tree = KDTree(max_depth=16, learning_rate=0.001, min_splitting_volume=0.00001,
                                 min_bounds=[0, -60], max_bounds=[20, 60],
                                 depth=0, sparse_vectors_file="./data/64k_sparse_vectors_seed_0.npz")
        cls.test_1_data = np.append(np.random.uniform(low=[0, -60], high=[20, 60], size=[64, 2]),
                                    np.random.multivariate_normal(mean=[10, 0], cov=[[5, 0], [0, 30]], size=64),
                                    axis=0)
        cls.test_1_statistics = {"number_of_encoding_points": 0.0,
                                 "elapsed_encoding_time": 0.0,
                                 "average_encoding_time": 0.0,
                                 "number_of_decoding_points": 0.0,
                                 "elapsed_decoding_time": 0.0,
                                 "average_decoding_time": 0.0,
                                 "rms_decoding_error": [0.0, 0.0]}

        cls.test_2_tree = KDTree(max_depth=16, learning_rate=0.001, min_splitting_volume=0.00001,
                                 min_bounds=[-10], max_bounds=[10],
                                 depth=0, sparse_vectors_file="./data/64k_sparse_vectors_seed_0.npz")
        cls.test_2_data = np.random.uniform(low=-10, high=10, size=[64, 1])
        cls.test_2_statistics = {"number_of_encoding_points": 0.0,
                                 "elapsed_encoding_time": 0.0,
                                 "average_encoding_time": 0.0,
                                 "number_of_decoding_points": 0.0,
                                 "elapsed_decoding_time": 0.0,
                                 "average_decoding_time": 0.0,
                                 "rms_decoding_error": [0.0, 0.0]}

    @classmethod
    def tearDownClass(cls):
        """
        Tear down method: print test statistics.
        """
        # Cartesian data
        print("Encoding/decoding Cartesian data:")
        print('Elapsed time to encode {0} points: {1:.2e} seconds'.format(
            cls.test_0_statistics["number_of_encoding_points"],
            cls.test_0_statistics["elapsed_encoding_time"]))
        print('Average encoding time: {:.2e} seconds'.format(cls.test_0_statistics["average_encoding_time"]))

        print('Elapsed time to decode {0} points: {1:.2e} seconds'.format(
            cls.test_0_statistics["number_of_decoding_points"],
            cls.test_0_statistics["elapsed_decoding_time"]))
        print('Average decoding time: {:.2e} seconds'.format(cls.test_0_statistics["average_decoding_time"]))
        print('Root mean squared decoding error (x-axis): {:.2e}'.format(cls.test_0_statistics["rms_decoding_error"][0]))
        print('Root mean squared decoding error (y-axis): {:.2e}'.format(cls.test_0_statistics["rms_decoding_error"][1]))

        # Polar data
        print("Encoding/decoding polar data:")
        print('Elapsed time to encode {0} points: {1:.2e} seconds'.format(
            cls.test_1_statistics["number_of_encoding_points"],
            cls.test_1_statistics["elapsed_encoding_time"]))
        print('Average encoding time: {:.2e} seconds'.format(cls.test_1_statistics["average_encoding_time"]))

        print('Elapsed time to decode {0} points: {1:.2e} seconds'.format(
            cls.test_1_statistics["number_of_decoding_points"],
            cls.test_1_statistics["elapsed_decoding_time"]))
        print('Average decoding time: {:.2e} seconds'.format(cls.test_1_statistics["average_decoding_time"]))
        print('Root mean squared decoding error (range): {:.2e}'.format(cls.test_1_statistics["rms_decoding_error"][0]))
        print('Root mean squared decoding error (azimuth): {:.2e}'.format(cls.test_1_statistics["rms_decoding_error"][1]))

        # Univariate data
        print("Encoding/decoding univariate data:")
        print('Elapsed time to encode {0} points: {1:.2e} seconds'.format(
            cls.test_2_statistics["number_of_encoding_points"],
            cls.test_2_statistics["elapsed_encoding_time"]))
        print('Average encoding time: {:.2e} seconds'.format(cls.test_2_statistics["average_encoding_time"]))

        print('Elapsed time to decode {0} points: {1:.2e} seconds'.format(
            cls.test_2_statistics["number_of_decoding_points"],
            cls.test_2_statistics["elapsed_decoding_time"]))
        print('Average decoding time: {:.2e} seconds'.format(cls.test_2_statistics["average_decoding_time"]))
        print('Root mean squared decoding error: {:.2e}'.format(cls.test_2_statistics["rms_decoding_error"][0]))

        cls.test_0_tree = None
        cls.test_0_data = None
        cls.test_0_statistics = None
        cls.test_1_tree = None
        cls.test_1_data = None
        cls.test_1_statistics = None
        cls.test_2_tree = None
        cls.test_2_data = None
        cls.test_2_statistics = None

    def test_0_codec(self):
        """
        Test case 0: batch encoding/decoding Cartesian data.
        """
        np.random.seed(0)
        exec_codec_test(self.test_0_tree, self.test_0_data, self.test_0_statistics, style=0, dx=0, dy=1)

    def test_1_codec(self):
        """
        Test case 1: batch encoding/decoding polar data.
        """
        np.random.seed(0)
        exec_codec_test(self.test_1_tree, self.test_1_data, self.test_1_statistics, style=1, dx=0, dy=1)

    def test_2_codec(self):
        """
        Test case 1: batch encoding/decoding univariate data.
        """
        np.random.seed(0)
        exec_codec_test(self.test_2_tree, self.test_2_data, self.test_2_statistics, style=0, dx=0, dy=0)


if __name__ == '__main__':
    unittest.main()

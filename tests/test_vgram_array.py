import numpy as np
import unittest
import time
import sys
from wnnlib.vgram.VGRAMArray import VGRAMArray


class TestVGRAMArray(unittest.TestCase):
    """
    Extends unittest.TestCase class to implement unit tests for the VGRAMArray class.
    """
    output_dims = (16, 16)
    number_of_patterns = 256
    min_mem_size = 63
    max_mem_size = 64
    min_dist = 2
    max_dist = 8
    pattern_length = 16
    array = None
    test_statistics = None

    @classmethod
    def setUpClass(cls):
        """
        Set up method: configure parameters and create a VGRAM node.
        """
        cls.array = VGRAMArray(output_dims=cls.output_dims, pattern_length=cls.pattern_length,
                               min_mem_size=cls.min_mem_size, max_mem_size=cls.max_mem_size,
                               min_learn_dist=cls.min_dist, max_recall_dist=cls.max_dist,
                               default_outputs=np.zeros(shape=cls.output_dims, order='C', dtype=np.int32), type_outputs=np.int32)
        cls.test_statistics = {"average_recall_time": 0.0,
                               "average_learn_time": 0.0,
                               "elapsed_recall_time": 0.0,
                               "elapsed_learn_time": 0.0}

    @classmethod
    def tearDownClass(cls):
        """
        Tear down method: print test statistics.
        """
        print('Elapsed time to learn {0} patterns: {1:.2e} seconds'.format(cls.number_of_patterns,
                                                                           cls.test_statistics["elapsed_learn_time"]))
        print('Average learn time: {:.2e} seconds'.format(cls.test_statistics["average_learn_time"]))
        print('Elapsed time to recall {0} patterns: {1:.2e} seconds'.format(cls.number_of_patterns,
                                                                            cls.test_statistics["elapsed_recall_time"]))
        print('Average recall time: {:.2e} seconds'.format(cls.test_statistics["average_recall_time"]))
        print('Average learn-recall time ratio: {:.2e} '.format(cls.test_statistics["average_learn_time"] /
                                                                cls.test_statistics["average_recall_time"]))
        cls.node = None
        cls.test_statistics = None

    def test_0_learn(self):
        """
        Test case 0: batch learning.
        """
        elapsed_time = 0
        for n in range(0, self.number_of_patterns):
            pattern = np.random.randint(low=0, high=2, size=self.pattern_length, dtype=bool)
            output_steps = np.random.randint(low=0, high=2, size=self.output_dims, dtype=int)
            t = time.time()
            self.array.learn(pattern, output_steps)
            elapsed_time += time.time() - t
        self.test_statistics["elapsed_learn_time"] = elapsed_time
        self.test_statistics["average_learn_time"] = elapsed_time / self.number_of_patterns

    def test_1_recall(self):
        """
        Test case 1: batch recalling.
        """
        elapsed_time = 0
        for n in range(0, self.number_of_patterns):
            pattern = np.random.randint(low=0, high=2, size=self.pattern_length, dtype=bool)
            t = time.time()
            values = self.array.recall(pattern)
            elapsed_time += time.time() - t
            if not "pytest" in sys.modules:
                self.array.debug()
        self.test_statistics["elapsed_recall_time"] = elapsed_time
        self.test_statistics["average_recall_time"] = elapsed_time / self.number_of_patterns


if __name__ == '__main__':
    unittest.main()

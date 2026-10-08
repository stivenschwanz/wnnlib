import numpy as np
import unittest
import time
import sys
from wnnlib.vgram.VGRAMNode import VGRAMNode


class TestVGRAMNode(unittest.TestCase):
    """
    Extends unittest.TestCase class to implement unit tests for the VGRAM class.
    """
    number_of_patterns = 512
    min_mem_size = 63
    max_mem_size = 64
    min_learn_dist = 2
    max_recall_dist = 8
    pattern_length = 16
    node = None
    test_statistics = None

    @classmethod
    def setUpClass(cls):
        """
        Set up method: configure parameters and create a VGRAM node.
        """
        cls.node = VGRAMNode(pattern_length=cls.pattern_length,
                             min_mem_size=cls.min_mem_size, max_mem_size=cls.max_mem_size,
                             min_learn_dist=cls.min_learn_dist, max_recall_dist=cls.max_recall_dist)
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
        elapsed_time = 0.0
        for n in range(0, self.number_of_patterns):
            pattern = np.random.randint(low=0, high=2, size=self.pattern_length, dtype=bool)
            t = time.time()
            self.node.learn(pattern, 1)
            elapsed_time += time.time() - t
            if not "pytest" in sys.modules:
                self.node.debug(pattern)
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
            value, _, _ = self.node.recall(pattern)
            if not "pytest" in sys.modules:
                self.node.debug(pattern, value)
            elapsed_time += time.time() - t
        self.test_statistics["elapsed_recall_time"] = elapsed_time
        self.test_statistics["average_recall_time"] = elapsed_time / self.number_of_patterns


if __name__ == '__main__':
    unittest.main()

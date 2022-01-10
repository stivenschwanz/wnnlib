from numpy import random
import numpy as np
from matplotlib import pyplot as plt
import unittest
import time
from wnnlib import VGRAMNode


class VGRAMLayer:
    """
    This class defines a layer of Virtual Generalized Random Access Memory (VGRAM) nodes.
    """
    def __init__(self, output_dims, pattern_length, min_mem_size, max_mem_size, min_dist, max_dist):
        """
            Initialize an empty VGRAM node.

            Parameters:
                output_dims (int, int): Output dimensions.
                pattern_length (int): Length of the stored patterns.
                min_mem_size (int): Minimum memory size.
                max_mem_size (int): Maximum memory size.
                min_dist (int): Minimum Hamming distance.
                max_dist (int): Maximum Hamming distance.
        """
        # Initialize layer nodes
        self.nodes = np.empty(output_dims, dtype=object)
        it = np.nditer(self.nodes, flags=['multi_index', 'refs_ok'], op_flags=['readwrite'])
        while not it.finished:
            self.nodes[it.multi_index] = VGRAMNode.VGRAMNode(pattern_length=pattern_length,
                                                             min_mem_size=min_mem_size, max_mem_size=max_mem_size,
                                                             min_dist=min_dist, max_dist=max_dist)
            it.iternext()

        # Initialize layer outputs
        self.output_values = np.zeros(output_dims, dtype=int)

        # Debug options
        self.fig = None
        self.ax = None

    def recall(self, input_pattern):
        """
        Recall the output value associated with the closest stored pattern to the input pattern.

        Parameters:
            input_pattern (bool[]): Input pattern.

        Returns:
            (int[]): Output values.
        """
        it = np.nditer(self.output_values, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            self.output_values[it.multi_index] = self.nodes[it.multi_index].recall(input_pattern)
            it.iternext()

        return self.output_values

    def learn(self, input_pattern, output_steps):
        """
        Update the stored value associated with the closest input patter to an input pattern.

        Parameters:
            input_pattern (bool[]): Input pattern.
            output_steps (int[]): Output steps.
        """
        it = np.nditer(self.nodes, flags=['multi_index', 'refs_ok'], op_flags=['readwrite'])
        while not it.finished:
            self.nodes[it.multi_index].learn(input_pattern, output_steps[it.multi_index])
            it.iternext()

    def debug(self):
        """
        Debug layer outputs.
        """
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            self.ax.axes.xaxis.set_visible(False)
            self.ax.axes.yaxis.set_visible(False)

        self.ax.clear()
        self.ax.imshow(self.output_values, cmap='gray', vmin=0, vmax=15, interpolation='nearest')
        plt.show(block=False)
        plt.pause(0.00001)


class TestVGRAMLayer(unittest.TestCase):
    """
    Extends unittest.TestCase class to implement unit tests for the VGRAMLayer class.
    """
    output_dims = (16, 16)
    number_of_patterns = 256
    min_mem_size = 63
    max_mem_size = 64
    min_dist = 2
    max_dist = 8
    pattern_length = 16
    layer = VGRAMLayer(output_dims=output_dims, pattern_length=pattern_length,
                       min_mem_size=min_mem_size, max_mem_size=max_mem_size,
                       min_dist=min_dist, max_dist=max_dist)

    def test_0_learn(self):
        """
        Test case 0: batch learning.
        """
        elapsed = 0
        print('Learning %d patterns.' % self.number_of_patterns)
        for n in range(0, self.number_of_patterns):
            pattern = np.random.randint(low=0, high=2, size=self.pattern_length, dtype=bool)
            output_steps = np.random.randint(low=0, high=2, size=self.output_dims, dtype=bool)
            t = time.time()
            self.layer.learn(pattern, output_steps)
            elapsed += time.time() - t
        print('Elapsed time %s s' % elapsed)
        print('Average learn time %s s' % (elapsed / self.number_of_patterns))

    def test_1_recall(self):
        """
        Test case 1: batch recalling.
        """
        elapsed = 0
        print('Recalling %d patterns.' % self.number_of_patterns)
        for n in range(0, self.number_of_patterns):
            pattern = np.random.randint(low=0, high=2, size=self.pattern_length, dtype=bool)
            t = time.time()
            values = self.layer.recall(pattern)
            elapsed += time.time() - t
            self.layer.debug()
        print('Elapsed time %s s' % elapsed)
        print('Average recall time %s s' % (elapsed / self.number_of_patterns))


if __name__ == '__main__':
    unittest.main()

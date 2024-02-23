from numpy import random
import numpy as np
from matplotlib import pyplot as plt
import unittest
import time
from wnnlib.vgram_array import VGRAMNode


class VGRAMArray:
    """
    This class defines an array of Virtual Generalized Random Access Memory (VGRAM) nodes.
    """
    def __init__(self, output_dims, pattern_length, min_mem_size, max_mem_size, min_learn_dist, max_recall_dist, default_outputs):
        """
        Initialize an empty VGRAM node.

        Parameters:
            output_dims (int, int): Output dimensions.
            pattern_length (int): Length of the stored patterns.
            min_mem_size (int): Minimum memory size.
            max_mem_size (int): Maximum memory size.
            min_learn_dist (int): Minimum Hamming distance.
            max_recall_dist (int): Maximum Hamming distance.
            default_outputs (float[]): Default output values
        """
        # Sanity check
        if np.shape(default_outputs) is not output_dims:
            default_outputs = np.reshape(default_outputs, output_dims)

        # Initialize array nodes
        self.nodes = np.empty(output_dims, dtype=object)
        it = np.nditer(self.nodes, flags=['multi_index', 'refs_ok'], op_flags=['readwrite'])
        while not it.finished:
            self.nodes[it.multi_index] = VGRAMNode.VGRAMNode(pattern_length=pattern_length,
                                                             min_mem_size=min_mem_size,
                                                             max_mem_size=max_mem_size,
                                                             min_learn_dist=min_learn_dist,
                                                             max_recall_dist=max_recall_dist,
                                                             default_output=default_outputs[it.multi_index])
            it.iternext()

        # Initialize array outputs
        self.output_values = np.zeros(output_dims, order='C', dtype=float)

        # Adjust debug window dimensions if necessary
        aspect_ratio = output_dims[0]/output_dims[1]
        if 0.5 <= aspect_ratio <= 2:
            self.debug_dims = output_dims
        else:
            number_outputs = np.prod(output_dims)
            k = np.log2(number_outputs)
            kx = int(np.ceil(k/2))
            ky = int(np.floor(k/2))
            self.debug_dims = (2**kx, 2**ky)

        # Debug options
        self.fig = None
        self.ax = None

    def __del__(self):
        """
        Delete method.
        """
        self.cleanup()

    def cleanup(self):
        """
        Clean everything up.
        """
        if self.fig is not None:
            self.ax.clear()
            plt.close(self.fig)
            self.fig.canvas.manager.window.destroy()
            self.ax = None
            self.fig = None

        if self.nodes is not None:
            self.nodes = None

        if self.output_values is not None:
            self.output_values = None

    def recall(self, input_pattern):
        """
        Recall the output value associated with the closest stored pattern to the input pattern.

        Parameters:
            input_pattern (bool[]): Input pattern.

        Returns:
            (float[]): Output values.
        """
        it = np.nditer(self.output_values, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            self.output_values[it.multi_index] = self.nodes[it.multi_index].recall(input_pattern)
            it.iternext()

        return self.output_values

    def learn(self, input_pattern, output_values):
        """
        Update the stored value associated with the closest input patter to an input pattern.

        Parameters:
            input_pattern (bool[]): Input pattern.
            output_values (float[]): Output values.
        """
        # Sanity check
        if np.shape(output_values) is not self.output_values.shape:
            output_values = np.reshape(output_values, self.output_values.shape)

        it = np.nditer(self.nodes, flags=['multi_index', 'refs_ok'], op_flags=['readwrite'])
        while not it.finished:
            self.nodes[it.multi_index].learn(input_pattern, output_values[it.multi_index])
            it.iternext()

    def memory_stats(self):
        """
        Array memory statistics.

        Return:
            (float): array memory size (MB)
            (float): array memory capacity (MB)
            (float): array memory usage (%)
        """
        it = np.nditer(self.output_values, flags=['multi_index'], op_flags=['readwrite'])
        arr_mem_size_megabytes = 0
        arr_mem_capacity_megabytes = 0
        while not it.finished:
            node_mem_size_kilobytes, node_mem_capacity_kilobytes, _ = self.nodes[it.multi_index].memory_stats()
            arr_mem_size_megabytes += node_mem_size_kilobytes
            arr_mem_capacity_megabytes += node_mem_capacity_kilobytes
            it.iternext()
        arr_mem_size_megabytes /= 1024
        arr_mem_capacity_megabytes /= 1024
        arr_mem_usage_percent = arr_mem_size_megabytes/arr_mem_capacity_megabytes
        return arr_mem_size_megabytes, arr_mem_capacity_megabytes, arr_mem_usage_percent

    def debug(self):
        """
        Debug array outputs.
        """
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            self.ax.axes.xaxis.set_visible(False)
            self.ax.axes.yaxis.set_visible(False)
            # self.fig.canvas.mpl_connect('close_event', lambda _: self.fig.canvas.manager.window.destroy())
            # win = self.fig.canvas.manager.window
            # win.overrideredirect(1)  # draws a completely frameless window

        self.ax.clear()
        self.ax.imshow(self.output_values.reshape(self.debug_dims), cmap='gray', vmin=0, vmax=15, interpolation='nearest', aspect='auto')

        self.fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.00001)


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
                               min_dist=cls.min_dist, max_dist=cls.max_dist)
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
            self.array.debug()
        self.test_statistics["elapsed_recall_time"] = elapsed_time
        self.test_statistics["average_recall_time"] = elapsed_time / self.number_of_patterns


if __name__ == '__main__':
    unittest.main()

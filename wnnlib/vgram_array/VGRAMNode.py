from numpy import random
import numpy as np
from matplotlib import pyplot as plt
import unittest
import time


class VGRAMNode:
    """
    This class defines a Virtual Generalized Random Access Memory (VGRAM) node.
    """
    def __init__(self, pattern_length, min_mem_size, max_mem_size, min_learn_dist, max_recall_dist, default_output=None):
        """
        Initialize an empty VGRAM node.

        Parameters:
            pattern_length (int): Length of the stored patterns.
            min_mem_size (int): Minimum memory size.
            max_mem_size (int): Maximum memory size.
            min_learn_dist (int): Minimum Hamming distance.
            max_recall_dist (int): Maximum Hamming distance.
            default_output (float): Default output value
        """
        # Check pattern length
        assert pattern_length > 0

        # Check minimum and maximum memory sizes
        assert 0 < min_mem_size < max_mem_size

        # Check minimum and maximum Hamming distances
        assert 0 <= min_learn_dist <= max_recall_dist

        # Initialize constant members
        self.pattern_length = pattern_length
        self.min_mem_size = min_mem_size
        self.max_mem_size = max_mem_size
        self.min_learn_dist = min_learn_dist
        self.max_recall_dist = max_recall_dist
        self.default_output = default_output

        # Initialize memory
        self.input_patterns = np.zeros((max_mem_size, pattern_length), order='C', dtype=bool)
        self.output_values = np.zeros(max_mem_size, order='C', dtype=float)
        self.valid_pairs = np.zeros(max_mem_size, order='C', dtype=bool)
        self.num_valid_pairs = int(0)
        self.all_indexes = np.array(range(0, self.max_mem_size))

        # Debug options
        self.fig = None
        self.axs = None

    def find_closest_pattern(self, input_pattern):
        """
        Find the closest stored input pattern according to the Hamming distance.

        Parameters:
            input_pattern (bool[]): Array of booleans.

        Returns:
            (int, int): Hamming distance to the closest pattern and index of the closest pattern.
        """
        closest_pattern_dist = np.inf
        closest_pattern_idx = None
        valid_indexes = self.all_indexes[self.valid_pairs]
        for idx in valid_indexes:
            # Efficiently compute the Hamming distance between the patterns
            dist = np.count_nonzero(self.input_patterns[idx, :] != input_pattern)
            # Randomly update the closest pattern index if the stored pattern is at the minimum distance
            if dist < closest_pattern_dist or dist == closest_pattern_dist and np.random.randint(low=0, high=2) == 1:
                # Update the closest pattern distance
                closest_pattern_dist = dist
                closest_pattern_idx = idx
        return [closest_pattern_dist, closest_pattern_idx]

    def get_pattern_by_index(self, index):
        """
        Get the stored input pattern using its index.

        Parameters:
            index (int): Index of the pattern.

        Returns:
            (bool[]): Input patter at the given memory location.
        """
        input_pattern = None
        valid_indexes = self.all_indexes[self.valid_pairs]
        if index in valid_indexes:
            input_pattern = self.input_patterns[index, :]
        return input_pattern

    def recall(self, input_pattern):
        """
        Recall the output value associated with the closest stored pattern to the input pattern.

        Parameters:
            input_pattern (bool[]): Input pattern.

        Returns:
            (float): Output value.
        """
        # Default output value
        output_value = self.default_output

        # Find the closest stored pattern
        [closest_pattern_dist, closest_pattern_idx] = self.find_closest_pattern(input_pattern)

        if closest_pattern_dist <= self.max_recall_dist:
            # Return the output associated with the closest input pattern
            output_value = self.output_values[closest_pattern_idx]

        return output_value

    def learn(self, input_pattern, output_value):
        """
        Update the stored value associated with the closest input pattern to an input pattern.

        Parameters:
            input_pattern (bool[]): Input pattern.
            output_value (float): Output value.

        Return:
            (int): Index of the updated input-output pair
        """
        if self.default_output >= output_value:
            return None

        # Prune a low frequency pairs first
        if self.num_valid_pairs == self.max_mem_size:
            min_value = np.min(self.output_values)
            min_indexes = np.where(self.output_values == min_value)[0]
            np.random.shuffle(min_indexes)
            max_pruning = self.max_mem_size - self.min_mem_size
            prune_indexes = min_indexes[:max_pruning]
            self.input_patterns[prune_indexes, :] = np.zeros(self.pattern_length, order='C', dtype=bool)
            self.output_values[prune_indexes] = self.default_output
            self.valid_pairs[prune_indexes] = False
            self.num_valid_pairs -= len(prune_indexes)

        if self.num_valid_pairs == np.uint(0):
            # Store the first input - output pair
            self.input_patterns[0, :] = input_pattern
            self.output_values[0] = float(output_value)
            self.valid_pairs[0] = True
            self.num_valid_pairs = int(1)
            return 0
        else:
            # Find the closest stored pattern
            [closest_pattern_dist, closest_pattern_idx] = self.find_closest_pattern(input_pattern)

            if closest_pattern_dist <= self.min_learn_dist:
                # Update an existing input - output pair
                self.output_values[closest_pattern_idx] = float(output_value)
                return closest_pattern_idx
            else:
                # Store a new input - output pair
                empty_entry_idx = np.argmin(self.valid_pairs)
                self.input_patterns[empty_entry_idx, :] = input_pattern
                self.output_values[empty_entry_idx] = float(output_value)
                self.valid_pairs[empty_entry_idx] = True
                self.num_valid_pairs += int(1)
                return empty_entry_idx

    def memory_stats(self):
        """
        Node memory statistics.

        Return:
            (float): node memory size (KB)
            (float): node memory capacity (KB)
            (float): node memory usage (%)
        """
        pair_mem_size_bytes = float(self.pattern_length + 4)
        node_mem_size_kilobytes = self.num_valid_pairs * pair_mem_size_bytes / 1024
        node_mem_capacity_kilobytes = self.max_mem_size * pair_mem_size_bytes / 1024
        node_mem_usage_percent = node_mem_size_kilobytes/node_mem_capacity_kilobytes
        return node_mem_size_kilobytes, node_mem_capacity_kilobytes, node_mem_usage_percent

    def debug(self, input_pattern=None, output_value=None):
        """
        Debug node memory.
        """
        if self.fig is None:
            self.fig, self.axs = plt.subplots(2, 2, gridspec_kw={'width_ratios': [15, 1], 'height_ratios': [15, 1]})
            self.axs[0, 0].axes.xaxis.set_visible(False)
            self.axs[0, 0].axes.yaxis.set_visible(False)
            self.axs[0, 1].axes.xaxis.set_visible(False)
            self.axs[0, 1].axes.yaxis.set_visible(False)
            self.axs[1, 0].axes.xaxis.set_visible(False)
            self.axs[1, 0].axes.yaxis.set_visible(False)
            self.axs[1, 1].axes.xaxis.set_visible(False)
            self.axs[1, 1].axes.yaxis.set_visible(False)
        # Show stored patterns
        self.axs[0, 0].clear()
        self.axs[0, 0].imshow(self.input_patterns[self.valid_pairs, :], cmap='gray',
                              vmin=0, vmax=1, interpolation='nearest', aspect='auto')
        # Show stored output values
        self.axs[0, 1].clear()
        self.axs[0, 1].imshow(np.expand_dims(self.output_values[self.valid_pairs], axis=1), cmap='gray',
                              vmin=0, vmax=15, interpolation='nearest', aspect='auto')
        # Show input pattern (if given)
        self.axs[1, 0].clear()
        if input_pattern is not None:
            self.axs[1, 0].imshow(np.expand_dims(input_pattern, axis=0), cmap='gray',
                                  vmin=0, vmax=1, interpolation='nearest', aspect='auto')
        # Show output value (if given)
        self.axs[1, 1].clear()
        if output_value is not None:
            self.axs[1, 1].imshow(output_value*np.ones((1, 1), order='C', dtype=float), cmap='gray',
                                  vmin=0, vmax=15, interpolation='nearest', aspect='auto')
        self.fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.00001)


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
            value = self.node.recall(pattern)
            self.node.debug(pattern, value)
            elapsed_time += time.time() - t
        self.test_statistics["elapsed_recall_time"] = elapsed_time
        self.test_statistics["average_recall_time"] = elapsed_time / self.number_of_patterns


if __name__ == '__main__':
    unittest.main()

from numpy import random
import numpy as np
from matplotlib import pyplot as plt
import unittest
import time


class VGRAMNode:
    """This class defines a Virtual Generalized Random Access Memory (VGRAM) node."""
    def __init__(self, pattern_length, min_mem_size, max_mem_size, min_dist, max_dist):
        """
        Initialize an empty VGRAM node.

        Parameters:
            pattern_length (int): Length of the stored patterns.
            min_mem_size (int): Minimum memory size.
            max_mem_size (int): Maximum memory size.
            min_dist (int): Minimum Hamming distance.
            max_dist (int): Maximum Hamming distance.
        """
        # Check pattern length
        assert pattern_length > 0

        # Check minimum and maximum memory sizes
        assert 0 < min_mem_size < max_mem_size

        # Check minimum and maximum Hamming distances
        assert 0 <= min_dist <= max_dist

        # Initialize constant members
        self.pattern_length = pattern_length
        self.min_mem_size = min_mem_size
        self.max_mem_size = max_mem_size
        self.min_dist = min_dist
        self.max_dist = max_dist

        # Initialize memory
        self.input_patterns = np.zeros((max_mem_size, pattern_length), order='C', dtype=bool)
        self.output_values = np.zeros(max_mem_size, order='C', dtype=int)
        self.valid_pairs = np.zeros(max_mem_size, order='C', dtype=bool)
        self.num_valid_pairs = int(0)
        self.all_indexes = np.array(range(0, self.max_mem_size))

        # Debug options
        self.fig = None
        self.ax1 = None
        self.ax2 = None

    def find_closest_pattern(self, input_pattern):
        """
            Find the closest stored input pattern according to the Hamming distance.
            
            Parameters:
                input_pattern (bool[]): Array of booleans.
                
            Returns:
                (int, int): Hamming distance to the closest pattern and index of the closest pattern         
        """
        closest_pattern_dist = np.inf
        closest_pattern_idx = None
        valid_indexes = self.all_indexes[self.valid_pairs]
        for idx in valid_indexes:
            dist = np.count_nonzero(self.input_patterns[idx, :] != input_pattern)
            # Randomly update the closest pattern index if the stored pattern is at the minimum distance
            if dist < closest_pattern_dist or dist == closest_pattern_dist and np.random.randint(low=0, high=2) == 1:
                # Update the closest pattern distance
                closest_pattern_dist = dist
                closest_pattern_idx = idx
        return [closest_pattern_dist, closest_pattern_idx]

    def recall(self, input_pattern):
        """
            Recall the output value associated with the closest stored pattern to the input pattern.

            Parameters:
                input_pattern (bool[]): Input pattern.

            Returns:
                (int): Output value.
        """
        # Default to zero
        output_value = int(0)

        # Find the closest stored pattern
        [closest_pattern_dist, closest_pattern_idx] = self.find_closest_pattern(input_pattern)

        if closest_pattern_dist <= self.max_dist:
            # Return the output associated with the closest input pattern
            output_value = self.output_values[closest_pattern_idx]

        return output_value

    def learn(self, input_pattern, output_step=int(1)):
        """
            Update the stored value associated with the closest input patter to an input pattern.

            Parameters:
                input_pattern (bool[]): Input pattern.
                output_step (int): Output step.
        """
        # Skip learning
        if output_step == 0:
            return

        # Prune a low frequency pairs first
        if self.num_valid_pairs == self.max_mem_size:
            min_value = np.min(self.output_values)
            min_indexes = np.where(self.output_values == min_value)[0]
            np.random.shuffle(min_indexes)
            max_pruning = self.max_mem_size - self.min_mem_size
            prune_indexes = min_indexes[:max_pruning]
            self.input_patterns[prune_indexes, :] = np.zeros(self.pattern_length, order='C', dtype=bool)
            self.output_values[prune_indexes] = int(0)
            self.valid_pairs[prune_indexes] = False
            self.num_valid_pairs -= len(prune_indexes)

        if self.num_valid_pairs == np.uint(0):
            # Store the first input - output pair
            self.input_patterns[0, :] = input_pattern
            self.output_values[0] = int(1)
            self.valid_pairs[0] = True
            self.num_valid_pairs = int(1)
        else:
            # Find the closest stored pattern
            [closest_pattern_dist, closest_pattern_idx] = self.find_closest_pattern(input_pattern)

            if closest_pattern_dist <= self.min_dist:
                # Update an existing input - output pair
                self.output_values[closest_pattern_idx] += int(output_step)
            else:
                # Store a new input - output pair
                empty_entry_idx = np.argmin(self.valid_pairs)
                self.input_patterns[empty_entry_idx, :] = input_pattern
                self.output_values[empty_entry_idx] = int(output_step)
                self.valid_pairs[empty_entry_idx] = True
                self.num_valid_pairs += int(1)

    def debug(self):
        """
            Debug node memory.
        """
        if self.fig is None:
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
            self.ax1.axes.xaxis.set_visible(False)
            self.ax1.axes.yaxis.set_visible(False)
            self.ax2.axes.xaxis.set_visible(False)
            self.ax2.axes.yaxis.set_visible(False)
        self.ax1.clear()
        self.ax1.imshow(self.input_patterns, cmap='gray',
                        vmin=0, vmax=1, interpolation='nearest')
        self.ax2.clear()
        self.ax2.imshow(np.expand_dims(self.output_values, axis=0), cmap='gray',
                        vmin=0, vmax=15, interpolation='nearest')
        plt.show(block=False)
        plt.pause(0.00001)


class TestVGRAMNode(unittest.TestCase):
    """
       Extends unittest.TestCase class to implement unit tests for the VGRAM class.
    """
    number_of_patterns = 256
    min_mem_size = 63
    max_mem_size = 64
    min_dist = 2
    max_dist = 8
    pattern_length = 16
    node = VGRAMNode(pattern_length=pattern_length,
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
            t = time.time()
            self.node.learn(pattern)
            elapsed += time.time() - t
            self.node.debug()
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
            value = self.node.recall(pattern)
            print(value)
            elapsed += time.time() - t
        print('Elapsed time %s s' % elapsed)
        print('Average recall time %s s' % (elapsed / self.number_of_patterns))


if __name__ == '__main__':
    unittest.main()

from numpy import random
import numpy as np
from matplotlib.patches import Wedge, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib import pyplot as plt


class KDTree:
    """This class defines a self-adapting kd-tree which incrementally build an encoding function directly from data.
    More precisely, as we travel from the root node to the corresponding leaf to encode a data point, we build a binary
    sequence by concatenating '0' or '1' when we turn to the left or to the right child, respectively. Moreover,
    we adjust the splitting point at each visited node using a convex combination rule based on a given learning rate
    between the current splitting point and the data point at the corresponding node dimension."""
    def __init__(self, max_depth, learning_rate, min_splitting_volume, min_bounds, max_bounds, depth=0):
        """Initialize an empty kd-tree node"""

        # Check if the maximum depth is strictly positive
        assert max_depth > 0

        # Check if the learning rate is between zero and one
        assert 0 < learning_rate < 1

        # Check if the minimum splitting volume is strictly positive
        assert min_splitting_volume > 0

        # Check if the depth is between 0 an the maximum depth
        assert 0 <= depth <= max_depth

        # Initialize constant members
        self.depth = depth
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_splitting_volume = min_splitting_volume

        # Compute sizes
        self.min_bounds = np.asarray(min_bounds, dtype=np.float64)
        self.max_bounds = np.asarray(max_bounds, dtype=np.float64)
        self.sizes = self.max_bounds - self.min_bounds

        # Get the number of dimensions
        k = len(self.sizes)

        # Select the splitting dimension
        self.split_dim = self.depth % k

        # Check if the sizes of all k dimensions are strictly positive
        assert np.alltrue(self.sizes > 0)

        # Compute volume
        self.volume = np.prod(self.sizes)

        # Check if the volume is greater or equal than the minimum allowed splitting volume
        # assert self.volume >= self.min_splitting_volume

        # Initialize members
        self.split_point = None
        self.left_child = None
        self.right_child = None

        # Debug options
        self.fig = None
        self.ax = None

    def update_child_bounds(self):
        # Sanity check
        if self.depth == self.max_depth:
            return

        # Sanity check
        if self.split_point is None:
            return

        # Check bounds consistency and reset splitting point
        if self.split_point < self.min_bounds[self.split_dim] or self.split_point > self.max_bounds[self.split_dim]:
            self.split_point = None
            self.left_child = None
            self.right_child = None
            return

        # Compute the left child bounds, size and volume
        left_child_min_bounds = self.min_bounds.copy()
        left_child_max_bounds = self.max_bounds.copy()
        left_child_max_bounds[self.split_dim] = self.split_point
        left_child_sizes = left_child_max_bounds - left_child_min_bounds
        left_child_volume = np.product(left_child_sizes)

        # Compute the right child bounds, size and volume
        right_child_min_bounds = self.min_bounds.copy()
        right_child_min_bounds[self.split_dim] = self.split_point
        right_child_max_bounds = self.max_bounds.copy()
        right_child_sizes = right_child_max_bounds - right_child_min_bounds
        right_child_volume = np.product(right_child_sizes)

        if self.left_child is None:
            if self.volume >= self.min_splitting_volume:
                self.left_child = KDTree(self.max_depth, self.learning_rate, self.min_splitting_volume,
                                         left_child_min_bounds, left_child_max_bounds, self.depth + 1)
        else:
            self.left_child.min_bounds = left_child_min_bounds
            self.left_child.max_bounds = left_child_max_bounds
            self.left_child.sizes = left_child_sizes
            self.left_child.volume = left_child_volume

        if self.right_child is None:
            if self.volume >= self.min_splitting_volume:
                self.right_child = KDTree(self.max_depth, self.learning_rate, self.min_splitting_volume,
                                          right_child_min_bounds, right_child_max_bounds, self.depth + 1)
        else:
            self.right_child.min_bounds = right_child_min_bounds
            self.right_child.max_bounds = right_child_max_bounds
            self.right_child.sizes = right_child_sizes
            self.right_child.volume = right_child_volume

    def encode(self, point):
        """Encode a k-dimensional point as a max_depth-length binary sequence using the kd-tree"""

        # Check if the maximum depth was achieved
        if self.depth >= self.max_depth:
            return []

        # Get the number of dimensions
        k = len(point)

        # Check if the point has the same number of dimensions of the given kd-tree bounds
        assert k == len(self.sizes)

        # Check if the given point is within the tree bounds
        assert np.alltrue(self.min_bounds <= point) and np.alltrue(point <= self.max_bounds)

        # Get the point value at the splitting dimension
        dim_value = point[self.split_dim]

        # Update the splitting value according to the learning rate
        if self.split_point is None:
            self.split_point = dim_value
        else:
            self.split_point = self.learning_rate * dim_value + (1 - self.learning_rate) * self.split_point

        # Update the child bounds according to the new splitting value
        self.update_child_bounds()

        if self.split_point is None:
            return np.random.randint(low=0, high=2, size=self.max_depth - self.depth)

        if dim_value < self.split_point:
            if self.left_child is not None:
                return np.append([0], self.left_child.encode(point))
            else:
                return np.append([0], np.random.randint(low=0, high=2, size=self.max_depth-self.depth-1))
        elif dim_value > self.split_point:
            if self.right_child is not None:
                return np.append([1], self.right_child.encode(point))
            else:
                return np.append([1], np.random.randint(low=0, high=2, size=self.max_depth-self.depth-1))
        else:
            return np.random.randint(low=0, high=2, size=self.max_depth-self.depth)

    def decode(self, code):

        # Get the number of dimensions
        k = len(self.sizes)

        # Check if the maximum depth was achieved
        if self.depth >= self.max_depth:
            return np.random.uniform(low=self.min_bounds, high=self.max_bounds, size=k)

        # Get the code at the current depth
        depth_code = code[self.depth]

        # Check if the code is binary
        assert depth_code in [0, 1]

        if depth_code == 0 and self.left_child is not None:
            return self.left_child.decode(code)
        elif depth_code == 1 and self.right_child is not None:
            return self.right_child.decode(code)
        else:
            # return np.append(self.min_bounds, self.max_bounds)
            return 0.5*(self.min_bounds+self.max_bounds)
            # return np.random.uniform(low=self.min_bounds, high=self.max_bounds, size=k)

    def draw_rectangle(self):
        """Recursively plot a visualization of the KD tree region"""

        patches = []
        colors = []

        # self.update_child_bounds()

        if self.left_child is not None:
            left_child_patches, left_child_colors = self.left_child.draw_rectangle()
            patches += left_child_patches
            colors += left_child_colors

        if self.right_child is not None:
            right_child_patches, right_child_colors = self.right_child.draw_rectangle()
            patches += right_child_patches
            colors += right_child_colors

        if len(patches) == 0:
            patches += [Rectangle(self.min_bounds, *self.sizes)]
            colors += [self.depth]

        return patches, colors

    def draw_tree(self, dims=[0, 1]):
        # Debug options
        if self.fig is None:
            with plt.ion():
                self.fig, self.ax = plt.subplots()

        #self.ax.cla()
        patches, colors = self.draw_rectangle()
        collection = PatchCollection(patches, cmap=plt.cm.get_cmap('gray'), alpha=0.75)
        collection.set_array(np.asarray(colors))
        collection.set_edgecolor('k')
        self.ax.add_collection(collection)
        # cbar = plt.colorbar(collection)
        # cbar.set_label('depth', rotation=90)
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        #self.fig.canvas.draw()
        plt.show(block=False)
        plt.pause(0.0001)

    def draw_sector(self):
        """Recursively plot a visualization of the KD tree region"""
        patches = []

        self.update_child_bounds()

        if self.left_child is not None:
            patches += self.left_child.draw_sector()

        if self.right_child is not None:
            patches += self.right_child.draw_sector()

        if len(patches) == 0:
            patches += [Wedge((0, 0), self.max_bounds[1], self.min_bounds[0], self.max_bounds[0], width=(self.max_bounds[1]-self.min_bounds[1]))]

        return patches

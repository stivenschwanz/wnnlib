import numpy as np
from matplotlib.patches import Wedge, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib import pyplot as plt
from wnnlib.codecs.SparseCodec import SparseCodec
from wnnlib.utils.BitUtils import BitUtils


class KDTree(SparseCodec):
    """
    This class defines a self-adapting kd-tree which incrementally builds an encoding function directly from data.
    More precisely, as we travel from the root node to the corresponding leaf to encode a data point, we build a binary
    sequence by concatenating '0' or '1' when we turn to the left or to the right child, respectively. Moreover,
    we adjust the splitting point at each visited node using a convex combination rule based on a given learning rate
    between the current splitting point and the data point at the corresponding node dimension.
    """
    def __init__(self, max_depth, learning_rate, min_splitting_volume,
                 min_bounds, max_bounds, depth=0, sparse_vectors_file=None):
        """
        Initialize an empty kd-tree node

        Parameters:
            max_depth (int): Maximum depth of the kd-tree.
            learning_rate (double): Learning rate.
            min_splitting_volume (double): Minimum splitting volume.
            min_bounds (double[]): Minimum bounds.
            max_bounds (double[]): Maximum bounds.
            depth (int): Subtree depth. Default is 0 for the root node.
            sparse_vectors_file (string): File containing the sparse vectors.
        """
        # Check if the maximum depth is strictly positive
        assert max_depth > 0

        # Check if the learning rate is between zero and one
        assert 0 <= learning_rate <= 1

        # Check if the minimum splitting volume is strictly positive
        assert min_splitting_volume > 0

        # Check if the depth is between 0 and the maximum depth
        assert 0 <= depth <= max_depth

        # Initialize constant members
        self.depth = depth
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_splitting_volume = min_splitting_volume

        # Compute sizes
        self.min_bounds = np.asarray(min_bounds, dtype=np.float64)*np.ones(1)
        self.max_bounds = np.asarray(max_bounds, dtype=np.float64)*np.ones(1)
        self.sizes = self.max_bounds - self.min_bounds

        # Get the number of dimensions
        k = len(self.sizes)

        # Select the splitting dimension
        self.split_dim = self.depth % k

        # Check if the sizes of all k dimensions are strictly positive
        assert np.all(self.sizes > 0)

        # Compute volume
        self.volume = np.prod(self.sizes)

        # Check if the volume is greater or equal than the minimum allowed splitting volume
        assert self.volume >= self.min_splitting_volume

        # Initialize members
        self.split_point = None
        self.left_child = None
        self.right_child = None

        # Debug options
        self.fig = None
        self.ax = None

        if sparse_vectors_file is not None:
            super().__init__(sparse_vectors_file)

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

        if self.left_child is not None:
            self.left_child = None

        if self.right_child is not None:
            self.right_child = None

    def get_split_value(self):
        return (1 - self.split_point) * self.min_bounds[self.split_dim] + \
               self.split_point * self.max_bounds[self.split_dim]

    def update_split_point(self, dim_value):
        if self.split_point is None:
            # Evenly splits the node at the first time (note that this is not exactly a kd-tree)
            # self.split_point = 0.5
            curr_point = (dim_value - self.min_bounds[self.split_dim]) / self.sizes[self.split_dim]
            # if curr_point < 0.5:
            #     #curr_point += 0.01 / self.sizes[self.split_dim]
            #     curr_point = (curr_point + 0.5)/2
            # elif curr_point > 0.5:
            #     #curr_point -= 0.01 / self.sizes[self.split_dim]
            #     curr_point = (curr_point + 0.5)/2
            curr_point = (curr_point + 0.5) / 2
            self.split_point = curr_point
        else:
            # Adjusts the existing splitting point a bit according to the learning rate
            curr_point = (dim_value - self.min_bounds[self.split_dim]) / self.sizes[self.split_dim]
            self.split_point = self.learning_rate * curr_point + (1 - self.learning_rate) * self.split_point

    def update_child_bounds(self):
        """
        Auxiliary method to update the subtree bounds.
        """
        # Sanity check
        if self.depth == self.max_depth:
            return

        # Sanity check
        if self.split_point is None:
            return

        # Sanity check
        if self.learning_rate == 0.0:
            return

        # Get splitting value
        split_value = self.get_split_value()

        # Compute the left child bounds, size and volume
        left_child_min_bounds = self.min_bounds.copy()
        left_child_max_bounds = self.max_bounds.copy()
        left_child_max_bounds[self.split_dim] = split_value
        left_child_sizes = left_child_max_bounds - left_child_min_bounds
        left_child_volume = np.prod(left_child_sizes)

        # Compute the right child bounds, size and volume
        right_child_min_bounds = self.min_bounds.copy()
        right_child_min_bounds[self.split_dim] = split_value
        right_child_max_bounds = self.max_bounds.copy()
        right_child_sizes = right_child_max_bounds - right_child_min_bounds
        right_child_volume = np.prod(right_child_sizes)

        if self.left_child is None:
            if left_child_volume >= self.min_splitting_volume:
                self.left_child = KDTree(self.max_depth, self.learning_rate, self.min_splitting_volume,
                                         left_child_min_bounds, left_child_max_bounds, self.depth + 1)
        else:
            self.left_child.min_bounds = left_child_min_bounds
            self.left_child.max_bounds = left_child_max_bounds
            self.left_child.sizes = left_child_sizes
            self.left_child.volume = left_child_volume
            self.left_child.update_child_bounds()

        if self.right_child is None:
            if right_child_volume >= self.min_splitting_volume:
                self.right_child = KDTree(self.max_depth, self.learning_rate, self.min_splitting_volume,
                                          right_child_min_bounds, right_child_max_bounds, self.depth + 1)
        else:
            self.right_child.min_bounds = right_child_min_bounds
            self.right_child.max_bounds = right_child_max_bounds
            self.right_child.sizes = right_child_sizes
            self.right_child.volume = right_child_volume
            self.right_child.update_child_bounds()

    def encode_point_into_binary_sequence(self, point):
        """
        Encode a k-dimensional point as a max_depth-length binary sequence using the kd-tree.

        Parameters:
            point (double[]): k-dimensional point.

        Returns:
            (int[]): Sparse binary representation of the given point.
        """

        # Check if the maximum depth was achieved
        if self.depth >= self.max_depth:
            return []

        point *= np.ones(1)

        # Get the number of dimensions
        k = len(point)

        # Check if the point has the same number of dimensions of the given kd-tree bounds
        assert k == len(self.sizes)

        # Check if the given point is within the tree bounds
        assert np.all(self.min_bounds <= point) and np.all(point <= self.max_bounds)

        # Get the point value at the splitting dimension
        dim_value = point[self.split_dim]

        # Update the splitting value according to the learning rate
        self.update_split_point(dim_value)

        # Update the child bounds according to the new splitting point
        self.update_child_bounds()

        # Get splitting value
        split_value = self.get_split_value()

        if dim_value < split_value:
            if self.left_child is not None:
                return np.append([np.uint8(0)], self.left_child.encode_point_into_binary_sequence(point))
            else:
                return np.zeros(self.max_depth-self.depth, order='C', dtype=np.uint8)
        elif dim_value > split_value:
            if self.right_child is not None:
                return np.append([np.uint8(1)], self.right_child.encode_point_into_binary_sequence(point))
            else:
                return np.append([np.uint8(1)], np.zeros(self.max_depth-self.depth-1, order='C', dtype=np.uint8))
        else:
            return np.zeros(self.max_depth-self.depth, order='C', dtype=np.uint8)

    def dense_vector_to_sparse_vector_index(self, dense_vector):
        """
        Compute the index of the sparse vector corresponding to a given dense vector.

        Parameters:
            dense_vector (float[]): an array lying in a compact feature/observation space.

        Returns:
            (int): Sparse vector index.
        """
        # Encode the given k-dimensional point into a binary sequence
        code = self.encode_point_into_binary_sequence(point=dense_vector)

        # Pack the code into a unique index
        sparse_vector_index = BitUtils.binary_array_to_integer(np.uint8(code))

        return sparse_vector_index

    def decode_binary_sequence_into_point(self, code):
        """
        Decode a sparse representation as a k-dimensional point using the kd-tree.

        Parameters:
            code (int[]): Sparse binary representation.

        Returns:
            (double[]): k-dimensional decoded point.
        """
        # Get the number of dimensions
        k = len(self.sizes)

        # Check if the maximum depth was achieved
        if self.depth >= self.max_depth:
            return 0.5 * (self.min_bounds + self.max_bounds)

        # Get the code at the current depth
        depth_code = code[self.depth]

        # Check if the code is binary
        assert depth_code in [0, 1]

        if depth_code == 0 and self.left_child is not None:
            return self.left_child.decode_binary_sequence_into_point(code)
        elif depth_code == 1 and self.right_child is not None:
            return self.right_child.decode_binary_sequence_into_point(code)
        else:
            return 0.5 * (self.min_bounds+self.max_bounds)

    def sparse_vector_index_to_dense_vector(self, sparse_vector_index):
        """
        Compute the dense vector corresponding to a given sparse vector index.

        Parameters:
            sparse_vector_index (int): Index of the sparse vector.

        Returns:
            (float[]): Array lying in a compact feature/observation space.
        """

        # Unpack the index into a unique code
        code = BitUtils.integer_to_binary_array(sparse_vector_index, self.max_depth)

        # Decode the given binary sequence into a k-dimensional point
        point = self.decode_binary_sequence_into_point(code)

        return point

    def draw_rectangle(self, dx=0, dy=1):
        """
        Recursively plot a visualization of the kd-tree using rectangles.
        """

        patches = []
        colors = []

        # self.update_child_bounds()

        if self.left_child is not None:
            left_child_patches, left_child_colors = self.left_child.draw_rectangle(dx, dy)
            patches += left_child_patches
            colors += left_child_colors

        if self.right_child is not None:
            right_child_patches, right_child_colors = self.right_child.draw_rectangle(dx, dy)
            patches += right_child_patches
            colors += right_child_colors

        if len(patches) == 0:
            patches += [Rectangle([self.min_bounds[dx], self.min_bounds[dy]], self.sizes[dx], self.sizes[dy])]
            colors += [self.depth]

        return patches, colors

    def draw_sector(self):
        """
        Recursively plot a visualization of the kd-tree using sectors.
        """
        patches = []
        colors = []

        self.update_child_bounds()

        if self.left_child is not None:
            left_child_patches, left_child_colors = self.left_child.draw_sector()
            patches += left_child_patches
            colors += left_child_colors

        if self.right_child is not None:
            right_child_patches, right_child_colors = self.right_child.draw_sector()
            patches += right_child_patches
            colors += right_child_colors

        if len(patches) == 0:
            patches += [Wedge((0, 0), self.max_bounds[0], self.min_bounds[1], self.max_bounds[1],
                              width=(self.max_bounds[0]-self.min_bounds[0]))]
            colors += [self.depth]

        return patches, colors

    def debug(self, style=0, dx=0, dy=1):
        """
        Recursively plot a visualization of the kd-tree .

        Parameters:
            style (int): Tree style (0: rectangles, 1: sectors)
            dx (int): Index of the dimension corresponding to the X axis.
            dy (int): Index of the dimension corresponding to the Y axis.
        """
        # Debug options
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            # win = self.fig.canvas.manager.window
            # win.overrideredirect(1)  # draws a completely frameless window

        # Clear axis
        self.ax.clear()
        self.ax.set_aspect('equal', adjustable='box')

        if style == 0:
            patches, colors = self.draw_rectangle(dx, dy)
            collection = PatchCollection(patches, cmap=plt.cm.get_cmap('gray'), ec='k', alpha=0.75)
            collection.set_array(np.asarray(colors))
            collection.set_edgecolor('k')
            self.ax.add_collection(collection)
            self.ax.set_xlim(self.min_bounds[dx], self.max_bounds[dx])
            self.ax.set_ylim(self.min_bounds[dy], self.max_bounds[dy])
        elif style == 1:
            circle = plt.Circle([0, 0], radius=self.max_bounds[dx], ec='k', fc='w')
            self.ax.add_patch(circle)
            patches, colors = self.draw_sector()
            collection = PatchCollection(patches, cmap=plt.cm.get_cmap('gray'), ec='k', alpha=0.75)
            collection.set_array(np.array(colors))
            self.ax.add_collection(collection)
            self.ax.set_xlim(-self.max_bounds[dx], self.max_bounds[dx])
            self.ax.set_ylim(-self.max_bounds[dx], self.max_bounds[dx])

        # self.ax.redraw_in_frame()
        self.fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.0001)

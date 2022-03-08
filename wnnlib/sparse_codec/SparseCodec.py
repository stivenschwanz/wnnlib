import unittest
from abc import ABC, abstractmethod
from numpy import random
import numpy as np
from wnnlib.BitUtils import BitUtils


class SparseCodec(ABC):
    """
    This abstract class defines a sparse encoder/decoder (codec).

    The sparse codec converts (encode) a dense vector lying in a compact feature/observation space into a sparse vector
    lying in a high-dimensional binary space. Conversely, it also converts (decode) a sparse vector into a dense vector.

    This class loads a set of pre-computed high-dimensional sparse vectors such that the following holds:
        a) The maximum number of activated bits of any sparse vector is 'maximum_number_of_activated_bits'.
        b) The minimum Hamming distance between any two sparse vectors is 'minimum_hamming_distance_between_vectors'.

    The subclasses must implement two methods:
        1) dense_vector_to_sparse_vector_index: compute the index of the sparse vector corresponding to a given dense
        vector.
        2) sparse_vector_index_to_dense_vector: compute the dense vector corresponding to a given sparse vector index.

    Note therefore that the subclasses must deal with the mapping between dense vectors and sparse vector indexes.
    On the other hand, the mapping between indexes and sparse vectors themselves are handled by the superclass.

    """

    def __init__(self, sparse_vectors_file):
        """
        Initialize a sparse codec.

        Loads a binary file containing a pre-computed set of high-dimensional sparse vectors.

        Parameters:
            sparse_vectors_file (string): name of the file containing the sparse vectors.
        """
        # Load the sparse vectors
        self._sparse_vectors, sparse_vector_parameters = SparseCodec.load_sparse_vectors(sparse_vectors_file)

        # Set sparse vector parameters
        [self._random_seed,
         self._number_of_sparse_vectors,
         self._sparse_vectors_length,
         self._maximum_number_of_activated_bits,
         self._minimum_hamming_distance_between_vectors] = sparse_vector_parameters

    @property
    def random_seed(self):
        """Get the random seed."""
        return self._random_seed

    @property
    def number_of_sparse_vectors(self):
        """Get the number of sparse vectors."""
        return self._number_of_sparse_vectors

    @property
    def sparse_vectors_length(self):
        """Get the length of the sparse vectors."""
        return self._sparse_vectors_length

    @property
    def maximum_number_of_activated_bits(self):
        """Get the maximum number of activated bits."""
        return self._maximum_number_of_activated_bits

    @property
    def minimum_hamming_distance_between_vectors(self):
        """Get the minimum Hamming distance between vectors."""
        return self._minimum_hamming_distance_between_vectors

    @abstractmethod
    def dense_vector_to_sparse_vector_index(self, dense_vector):
        """
        Compute the index of the sparse vector corresponding to a given dense vector.

        Parameters:
            dense_vector (float[]): an array lying in a compact feature/observation space.

        Returns:
            (int): Sparse vector index.
        """
        print("dense_vector_to_sparse_vector_index")
        pass

    def encode(self, dense_vector):
        """
        Encode a dense vector into a high-dimensional, sparse binary representation.

        Parameters:
            dense_vector (float[]): an array lying in a compact feature/observation space.

        Returns:
            (int[]): High-dimensional vector containing a sparse binary representation of the given dense vector.
        """
        # Compute the index of the sparse vector corresponding to a given dense vector
        sparse_vector_index = self.dense_vector_to_sparse_vector_index(dense_vector)

        # Check sparse vector index
        assert 0 <= sparse_vector_index < self._number_of_sparse_vectors

        # Get the sparse vector according to the selected index
        sparse_vector = self._sparse_vectors[sparse_vector_index]

        return sparse_vector

    def one_hot_encode(self, sparse_vector):
        """
        Encode a sparse vector into a high-dimensional, one-hot vector representation.

        Parameters:
            sparse_vector (int[]): Sparse binary representation.

        Returns:
            (int[]): High-dimensional vector containing a one-hot representation of the given dense vector.
        """
        # Find the index of the closest sparse vector to the given sparse vector
        sparse_vector_index = self.find_closest_sparse_vector(sparse_vector)

        # Check sparse vector index
        assert 0 <= sparse_vector_index < self._number_of_sparse_vectors

        # Create the one-hot vector
        one_hot_vector = np.zeros(self._number_of_sparse_vectors, order='C', dtype=np.uint8)
        one_hot_vector[sparse_vector_index] = np.uint8(1)

        return one_hot_vector

    @abstractmethod
    def sparse_vector_index_to_dense_vector(self, sparse_vector_index):
        """
        Compute the dense vector corresponding to a given sparse vector index.

        Parameters:
            sparse_vector_index (int): Index of the sparse vector.

        Returns:
            (float[]): Array lying in a compact feature/observation space.
        """
        print("sparse_vector_index_to_dense_vector")
        pass

    def find_closest_sparse_vector(self, sparse_vector):
        """
        Find the index of the closest stored sparse vector in terms of the Hamming distance to a given sparse vector.

        Parameters:
            sparse_vector (int[]): Given sparse vector.

        Returns:
            (int, int): The Hamming distance to and the index of the closest stored sparse vector.
        """
        closest_sparse_vector_dist = np.inf
        closest_sparse_vector_idx = None
        for idx in range(0, self._sparse_vectors.shape[0]):
            # Efficiently compute the Hamming distance between the sparse vectors
            dist = np.count_nonzero(self._sparse_vectors[idx, :] != sparse_vector)
            # Randomly update the closest sparse vector index if the stored sparse vector is at the minimum distance
            if dist < closest_sparse_vector_dist or \
                    dist == closest_sparse_vector_dist and \
                    np.random.randint(low=0, high=2) == 1:
                # Update the closest sparse vector distance
                closest_sparse_vector_dist = dist
                closest_sparse_vector_idx = idx
        return [closest_sparse_vector_dist, closest_sparse_vector_idx]

    def decode(self, sparse_vector):
        """
        Decode a sparse vector into a dense, low-dimensional vector.

        Parameters:
            sparse_vector (int[]): Sparse binary representation.

        Returns:
            (float[]): An array lying in a compact feature/observation space.
        """
        # Find the index of the closest stored sparse vector to a given sparse vector
        _, sparse_vector_index = self.find_closest_sparse_vector(sparse_vector)

        # Compute the dense vector according to the index of the corresponding sparse vector
        dense_vector = self.sparse_vector_index_to_dense_vector(sparse_vector_index)

        return dense_vector

    @staticmethod
    def generate_sparse_vectors(random_seed=0,
                                number_of_sparse_vectors=64 * 1024,
                                sparse_vectors_length=2048,
                                maximum_number_of_activated_bits=64,
                                minimum_hamming_distance_between_vectors=96,
                                output_file_name="./sparse_vectors.npz"):
        """
        Generate a consistent set of sparse vector and dump the vectors in a compressed .npz file.
        This method employs a brute force approach. Thus, there's a lot of room to improve it.

        Parameters:
            random_seed (int): Pseudo-random generator seed.
            number_of_sparse_vectors (int): Learning rate.
            sparse_vectors_length (int): Number of bits of each sparse vector.
            maximum_number_of_activated_bits (int): Maximum number of activated bits in each sparse vector.
            minimum_hamming_distance_between_vectors (int): Minimum Hamming distance between any two sparse vectors.
            output_file_name (string): Name of the output file.
        """

        # Compute the probability of activate each bit
        probability_of_activation = maximum_number_of_activated_bits / sparse_vectors_length

        # Initialize the random number seed
        np.random.seed(random_seed)

        # Allocate the vector to store the intermediate results
        sparse_vectors = np.zeros((number_of_sparse_vectors, sparse_vectors_length), order='C', dtype=np.uint8)

        n = 0
        while n < number_of_sparse_vectors:
            u = np.random.binomial(n=1, p=probability_of_activation, size=sparse_vectors_length)
            if np.sum(u) > maximum_number_of_activated_bits:
                continue

            min_hamming_distance = np.inf
            append_vector_flag = True
            m = 0
            while m < n:
                v = sparse_vectors[m, :]
                m += 1
                hamming_distance = BitUtils.hamming_distance(u, v)

                if hamming_distance < min_hamming_distance:
                    min_hamming_distance = hamming_distance

                if hamming_distance < minimum_hamming_distance_between_vectors:
                    append_vector_flag = False
                    break

            if append_vector_flag:
                # print("sum u %s", np.sum(u))
                # print("min dist u %s", min_dist)
                sparse_vectors[n, :] = u
                # print(n)
                n += 1

        # Sparse vector parameters
        sparse_vector_parameters = [random_seed,
                                    number_of_sparse_vectors,
                                    sparse_vectors_length,
                                    maximum_number_of_activated_bits,
                                    minimum_hamming_distance_between_vectors]

        # Dump the results to a file
        np.savez(output_file_name, sparse_vectors, sparse_vector_parameters)

    @staticmethod
    def load_sparse_vectors(input_file_name="sparse_vectors.npz"):
        """
        Load sparse vectors from a compressed .npz file

        Parameters:
            input_file_name (string): Name of the input file.

        Returns:
            (array, array): Arrays containing the sparse vectors and corresponding parameters.
        """
        npz_file = np.load(input_file_name)
        sparse_vectors = npz_file['arr_0']
        sparse_vector_parameters = npz_file['arr_1']
        return sparse_vectors, sparse_vector_parameters


class TestSparseCodec(unittest.TestCase):
    """
    Extends unittest.TestCase class to implement unit tests for the KDTree class.
    """

    class DummySparseCodec(SparseCodec):
        """
        Dummy class to test the sparse codec class.
        """

        def __init__(self, sparse_vectors_file):
            self.dense_vectors_to_indexes_dict = []
            self.counter = 0
            super(TestSparseCodec.DummySparseCodec, self).__init__(sparse_vectors_file=sparse_vectors_file)

        def dense_vector_to_sparse_vector_index(self, dense_vector):
            sparse_vector_index = self.counter
            self.counter += 1
            self.dense_vectors_to_indexes_dict.append(dense_vector)
            return sparse_vector_index

        def sparse_vector_index_to_dense_vector(self, sparse_vector_index):
            return self.dense_vectors_to_indexes_dict[sparse_vector_index]

    def test_0_codec(self):
        """
        Test case 0: generate the sparse vectors.
        """
        SparseCodec.generate_sparse_vectors(random_seed=0,
                                            number_of_sparse_vectors=2 * 1024,
                                            sparse_vectors_length=2048,
                                            maximum_number_of_activated_bits=64,
                                            minimum_hamming_distance_between_vectors=96,
                                            output_file_name="../../wnndata/2k_sparse_vectors_seed_0.npz")

    def test_1_codec(self):
        """
        Test case 1: load generated sparse vectors.
        """
        sparse_codec = TestSparseCodec.DummySparseCodec(sparse_vectors_file="../../wnndata/2k_sparse_vectors_seed_0.npz")

        dense_vector_length = 10
        dense_vector = np.random.random(size=dense_vector_length)
        print(dense_vector)
        sparse_vector = sparse_codec.encode(dense_vector)
        print(sparse_vector)
        dense_vector2 = sparse_codec.decode(sparse_vector)
        print(dense_vector2)
        one_hot_vector = sparse_codec.one_hot_encoder(dense_vector)
        print(one_hot_vector)

        print(getattr(sparse_codec, 'sparse_vectors_length'))


if __name__ == '__main__':
    unittest.main()

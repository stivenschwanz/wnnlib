import unittest
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors as clrs

class BitUtils:

    @staticmethod
    def hamming_distance(a, b):
        # return np.sum(np.bitwise_xor(a, b))
        return np.count_nonzero(a != b)

    @staticmethod
    def binary_array_to_integer(arr):
        val = np.int64(0)
        pow2 = np.int64(1)
        length = len(arr)
        for i in range(length-1, -1, -1):
            val += pow2 * arr[i]
            pow2 *= np.int64(2)
        return val

    @staticmethod
    def integer_to_binary_array(val, length):
        arr = np.zeros(length, order='C', dtype=np.uint8)
        i = length-1
        while val > 0 and i >= 0:
            arr[i] = val & 1
            val >>= 1
            i -= 1
        return arr

    @staticmethod
    def integer_to_binary_array2(val, length):
        arr = np.zeros(length, order='C', dtype=np.uint8)
        i = 0
        for bit in bin(val)[2:].zfill(length):
            arr[i] = np.uint8(bit)
            i += 1
        return arr

    @staticmethod
    def high(n):
        return np.ones(n, order='C', dtype=bool)

    @staticmethod
    def low(n):
        return np.zeros(n, order='C', dtype=bool)

    @staticmethod
    def sparse_vector(seq):
        vecs = []
        for (n, func) in seq:
            if n > 0:
                vecs.append(func(n))
        return np.concatenate(vecs)

    @staticmethod
    def mean_index(vec):
        return np.mean(np.nonzero(vec))

    @staticmethod
    def belief(c, p, alpha):
        """
        Build a belief representation from the hyper-parameters $ \\boldsymbol{\\alpha} $.

        Parameters:
            c (int): Number of categories.
            p (int): Maximum number of hypothesis to draw.
            alpha (float[]): Hyper-parameters $ \\boldsymbol{\\alpha} $.
        Return:
           (bool[]): Sparsely encoded belief $ {\\bf b} $.
           (int[]): Pseudo-counts vector $ {\\bf pc} $.
           (float[]): Probability distribution $ \\boldsymbol{\\pi} $.
       """

        # ----------------------------------------------------------------------
        # Draw the Categorical distribution $ \\boldsymbol{\\pi} $ from the Dirichlet
        # distribution $ Dir \\left( c; \\boldsymbol{\\alpha} \\right) $.
        # ----------------------------------------------------------------------
        pi = np.random.dirichlet(alpha)

        # ----------------------------------------------------------------------
        # Build the pseudo-counts vector $ {\\bf pc} $.
        # ----------------------------------------------------------------------
        pc = np.round(p * pi)

        # ----------------------------------------------------------------------
        # Build the sparsely encoded belief $ {\\bf b} $.
        # ----------------------------------------------------------------------
        acc = 0
        bel = np.zeros(c + p, order='C', dtype=bool)
        for i in np.arange(c + p):
            if i < c:
                acc += pc[i]
            if acc > 0:
                bel[i] = True
                acc -= 1

        return bel, pc, pi

    @staticmethod
    def unbelief(c, p, bel):
        """
        Build a probability distribution from the belief $ {\\bf b} $.

        Parameters:
            c (int): Number of categories.
            p (int): Maximum number of hypothesis to draw.
            bel (bool[]): Sparsely encoded belief $ {\\bf b} $.
        Return:
           (float[]): Probability distribution $ \\boldsymbol{\\pi} $.
       """

        # ----------------------------------------------------------------------
        # Build the sparsely encoded belief $ {\\bf b} $.
        # ----------------------------------------------------------------------
        acc = 0
        pi = np.zeros(c, order='C', dtype=float)
        for i in np.arange(c + p - 1, 1, -1):
            acc += float(bel[i])
            if i < c and (i == 1 or not bel[i - 1]):
                pi[i] = acc
                acc = 0
        pi = pi / np.sum(pi)

        return pi


class TestBitUtils(unittest.TestCase):
    """
    Extends unittest.TestCase class to implement unit tests for the VGRAMArray class.
    """

    """
        Extends unittest.TestCase class to implement unit tests for the VGRAMArray class.
        """
    number_of_patterns = 256
    pattern_length = 16
    test_statistics = None

    @classmethod
    def setUpClass(cls):
        """
        Set up method: configure parameters and create a VGRAM node.
        """
        cls.test_statistics = {"average_binary_array_to_integer_time": 0.0,
                               "average_integer_to_binary_array_time": 0.0,
                               "elapsed_binary_array_to_integer_time": 0.0,
                               "elapsed_integer_to_binary_array_time": 0.0}

    @classmethod
    def tearDownClass(cls):
        """
        Tear down method: print test statistics.
        """
        print('Elapsed time to convert {0} patterns: {1:.2e} seconds'.format(cls.number_of_patterns,
                                                                             cls.test_statistics["elapsed_binary_array_to_integer_time"]))
        print('Average convert time: {:.2e} seconds'.format(cls.test_statistics["average_binary_array_to_integer_time"]))
        print('Elapsed time to convert {0} patterns: {1:.2e} seconds'.format(cls.number_of_patterns,
                                                                             cls.test_statistics["elapsed_binary_array_to_integer_time"]))
        print('Average convert time: {:.2e} seconds'.format(cls.test_statistics["elapsed_integer_to_binary_array_time"]))

        cls.test_statistics = None

    def test_0(self):
        """
        Test case 0: binary array to integer.
        """
        arr = np.array([0, 0, 0, 0, 1, 0, 1], dtype=np.uint8)
        print(arr)
        val = BitUtils.binary_array_to_integer(arr)
        print(val)

    def test_1(self):
        """
        Test case 1: binary array to integer.
        """
        elapsed_time1 = 0
        elapsed_time2 = 0
        for n in range(0, self.number_of_patterns):
            arr1 = np.random.randint(low=0, high=2, size=self.pattern_length, dtype=np.uint8)
            t = time.time()
            val1 = BitUtils.binary_array_to_integer(arr1)
            elapsed_time1 += time.time() - t
            t = time.time()
            arr2 = BitUtils.integer_to_binary_array(val1, self.pattern_length)
            elapsed_time2 += time.time() - t
            val2 = BitUtils.binary_array_to_integer(arr2)
            assert val1 == val2

        self.test_statistics["elapsed_binary_array_to_integer_time"] = elapsed_time1
        self.test_statistics["average_binary_array_to_integer_time"] = elapsed_time1 / self.number_of_patterns
        self.test_statistics["elapsed_integer_to_binary_array_time"] = elapsed_time2
        self.test_statistics["average_integer_to_binary_array_time"] = elapsed_time2 / self.number_of_patterns

    def test_2(self):
        """
        Test case 2: Running example Steps 1 through 3 of the nP-GbF procedure.
        """

        # Hyper-parameters
        c = 24
        p = 8
        l1 = 4
        alpha1 = 2
        l2 = 12
        alpha2 = 1
        seed = 1
        l3 = 14
        alpha3 = 1
        seed = 0
        fontsize = 22

        np.random.seed(seed)

        # Dirichlet's hyperparameters initialization
        alpha0 = 2 ** -12
        alpha = alpha0*np.ones(c, order='C', dtype=float)
        alpha[l1] = alpha1
        alpha[l2] = alpha2
        alpha[l3] = alpha3

        # Build the belief from the Dirichlet hyperparameters
        bel, pc, pi = BitUtils.belief(c, p, alpha)

        # Recover the posterior
        rec_pi = BitUtils.unbelief(c, p, bel)

        fig = plt.figure(figsize=(16, 5/0.75))
        gs = fig.add_gridspec(4, hspace=0.5, height_ratios=[0.4, 0.25, 0.1, 0.25])
        axs = gs.subplots(sharex=True, sharey=False)

        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsthm}'

        axs[0].set_frame_on(True)
        axs[0].grid(True, linestyle='--', linewidth=1)
        axs[0].set_xlim(0.5, c+0.5)
        axs[0].set_ylim(0, np.max(alpha))
        axs[0].bar(np.arange(1, c+1, step=1), np.max(alpha)*np.ones(c, order='C', dtype=int), width=1, edgecolor='0.75', color='1.0')
        axs[0].bar(np.arange(1, c+1, step=1), alpha, width=1)
        axs[0].set_yticks(np.arange(0, np.max(alpha)+0.1, step=1))
        axs[0].set_title(r'{\bf Step 1}: Pseudo-counts $\boldsymbol{\alpha}_{n|n-1} \equiv \boldsymbol{\alpha}_{n|n-1}(\check{\bf z}_{n}, {\bf b}_{n|n-1})$', fontsize=fontsize, usetex=True)

        axs[1].set_frame_on(True)
        axs[1].grid(True, linestyle='--', linewidth=1)
        axs[1].set_xlim(0.5, c+0.5)
        axs[1].set_ylim(0, 1)
        axs[1].bar(np.arange(1, c+1, step=1), np.ones(c, order='C', dtype=int), width=1, edgecolor='0.75', color='1.0')
        axs[1].bar(np.arange(1, c+1, step=1), pi, width=1, edgecolor='0.75', facecolor='red')
        axs[1].set_yticks(np.arange(0, 1.1, step=0.5))
        axs[1].set_title(r'{\bf Step 2}: Sampled posterior $\boldsymbol{\pi}_{n+1|n} \sim Dir(c_{s}; \boldsymbol{\alpha}_{n|n-1}) $', fontsize=fontsize, usetex=True)

        axs[2].set_frame_on(True)
        axs[2].grid(True, linestyle='--', linewidth=1)
        axs[2].set_xlim(0.5, c+p+0.5)
        axs[2].set_ylim(0, 1)
        axs[2].bar(np.arange(1, c+p+1, step=1), np.ones(c+p, order='C', dtype=int), width=1, edgecolor='0.75',
                   color=(1.0*bel.astype(int)+0.0).astype(str))
        axs[2].set_title(r'{\bf Step 3}: Encoded belief ${\bf b}_{n+1|n} = Bel(\boldsymbol{\pi}_{n+1|n}; p_{s})$ ($\blacksquare=0$, $\square=1$)', fontsize=fontsize, usetex=True)
        axs[2].set_yticks([], [])

        axs[3].set_frame_on(True)
        axs[3].grid(True, linestyle='--', linewidth=1)
        axs[3].set_xlim(0.5, c+0.5)
        axs[3].set_ylim(0, 1)
        axs[3].bar(np.arange(1, c+1, step=1), np.ones(c, order='C', dtype=int), width=1, edgecolor='0.75', color='1.0')
        axs[3].bar(np.arange(1, c+1, step=1), rec_pi, width=1, edgecolor='0.5', facecolor='red')
        axs[3].set_yticks(np.arange(0, 1.1, step=0.5))
        axs[3].set_title(r'Recovered posterior $Bel^{-1}({\bf b}_{n+1|n}; p_{s})$', fontsize=fontsize, usetex=True)

        axs[3].set_xlabel(r'Symbol index $\ell^{\prime}$', fontsize=fontsize, usetex=True)
        axs[3].set_xticks([1, l1+1, l2+1, l3+1, c, c+p, c+p+0.5],
                          ['1', r'$\ell_{1}^{\prime}=$'+str(l1), r'$\ell_{2}^{\prime}=$'+str(l2),
                           r'$\ell_{3}^{\prime}=$'+str(l3), r'$c_{s}$', r'$c_{s}+p_{s}$', ''])

        plt.show(block=True)

    def test_3(self):
        """
        Test case 3: Running example Steps 4 through 6 of the nP-GbF procedure.
        """

        # Hyper-parameters
        c = 32
        p = 8
        l1 = 2
        alpha1 = 1
        l2 = 16
        alpha2 = 3
        seed = 1
        l3 = 24
        alpha3 = 1
        seed = 0
        fontsize = 22

        np.random.seed(seed)

        # Dirichlet's hyperparameters initialization
        alpha0 = 2 ** -12
        alpha = alpha0*np.ones(c, order='C', dtype=float)
        alpha[l1] = alpha1
        alpha[l2] = alpha2
        alpha[l3] = alpha3

        # Build the belief from the Dirichlet hyperparameters
        bel, pc, pi = BitUtils.belief(c, p, alpha)

        # Find the highest probability
        k_hat = np.argmax(pi)
        omega_z = np.zeros(c, order='C', dtype=int)
        omega_z[k_hat] = 1

        fig = plt.figure(figsize=(16, 5))
        gs = fig.add_gridspec(3, hspace=0.5, height_ratios=[0.4, 0.25, 0.1])
        axs = gs.subplots(sharex=True, sharey=False)

        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsthm}'

        axs[0].set_frame_on(True)
        axs[0].grid(True, linestyle='--', linewidth=1)
        axs[0].set_xlim(0.5, c+0.5)
        axs[0].set_ylim(0, np.max(alpha))
        axs[0].bar(np.arange(1, c+1, step=1), np.max(alpha)*np.ones(c, order='C', dtype=int), width=1, edgecolor='0.75', color='1.0')
        axs[0].bar(np.arange(1, c+1, step=1), alpha, width=1)
        axs[0].set_yticks(np.arange(0, np.max(alpha)+0.1, step=1))
        axs[0].set_title(r'{\bf Step 4}: Pseudo-counts $\tilde{\boldsymbol{\alpha}}_{n|n-1} \equiv \tilde{\boldsymbol{\alpha}}_{n|n-1}({\bf b}_{n+1|n})$', fontsize=fontsize, usetex=True)

        axs[1].set_frame_on(True)
        axs[1].grid(True, linestyle='--', linewidth=1)
        axs[1].set_xlim(0.5, c+0.5)
        axs[1].set_ylim(0, 1)
        axs[1].bar(np.arange(1, c+1, step=1), np.ones(c, order='C', dtype=int), width=1, edgecolor='0.75', color='1.0')
        axs[1].bar(np.arange(1, c+1, step=1), pi, width=1, edgecolor='0.75', facecolor='red')
        axs[1].set_yticks(np.arange(0, 1.1, step=0.5))
        axs[1].set_title(r'{\bf Step 5}: Sampled posterior $\tilde{\boldsymbol{\pi}}_{n+1|n} \sim Dir(c_{z}; \tilde{\boldsymbol{\alpha}}_{n|n-1}) $', fontsize=fontsize, usetex=True)

        axs[2].set_frame_on(True)
        axs[2].grid(True, linestyle='--', linewidth=1)
        axs[2].set_xlim(0.5, c+0.5)
        axs[2].set_ylim(0, 1)
        axs[2].bar(np.arange(1, c+1, step=1), np.ones(c, order='C', dtype=int), width=1, edgecolor='0.75', color=omega_z.astype(str))
        axs[2].set_title(r'{\bf Step 6}: Next observation $\hat{\bf z}_{n+1|n} = \boldsymbol{\mu}^{(\hat{k})}_{z}$ ($\blacksquare\neq\hat{k}$, $\square=\hat{k}$)', fontsize=fontsize, usetex=True)
        axs[2].set_yticks([], [])
        axs[2].set_xlabel(r'Symbol index $\ell$', fontsize=fontsize, usetex=True)
        axs[2].set_xticks([1, l1 + 1, l2 + 1, l3 + 1, c, c + 0.5],
                          ['1', r'$\ell_{1}=$' + str(l1), r'$\ell_{2}=$' + str(l2),
                           r'$\ell_{3}=$' + str(l3), r'$c_{z}$', ''])

        plt.show(block=True)


if __name__ == '__main__':
    unittest.main()

from numpy import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import unittest
import time
from wnnlib.vgram_array import VGRAMArray, VGRAMNode
from wnnlib.sparse_codec import ScalarCodec


class NPCLAD:
    """
    This class implements a continuous learning (CL), anomaly detector (AD) for streamed sequences of observations
    using a non-parametric Bayesian procedure to build a suitable model of the underlying stochastic process emitting
    the observations.
    """
    def __init__(self, cs, ps, alphas, az, bz, cz, dz, pz, alphaz, test, delta=0, learning_rate=1):
        """
        Initialize the nP-CLAD detector using the nine hyperparameters.

        Parameters:
            cs (int): Maximum number of distinct hidden state vectors (symbols) in $ \\boldsymbol{\\Omega}_{s} $.
            ps (int): Maximum number of equiprobable hypothesis in the belief $ {\\bf b}_{n|n-1} $, $ \\forall n > 0 $.
            alphas (float): Pseudo-count parameter for the prior Dirichlet distribution
                            $ Dir(c_{s}; \\boldsymbol{\\alpha}_{0}) $.
            az (int): Number of activated bits of the sparsely encoded observations.
            bz (int): Minimum distance between sparsely encoded observations.
            cz (int): Maximum number of distinct encoded observations (symbols) in $ \\boldsymbol{\\Omega}_{z} $.
            dz (int): Total number of bits of the sparsely encoded observations.
            pz (int): Maximum number of equiprobable hypothesis in the belief $ \\tilde{\\bf b}_{n|n-1} $,
                      $ \\forall n > 0 $.
            alphaz (float): Pseudo-count parameter for the prior Dirichlet distribution
                            $ Dir(c_{z}; \\tilde\\boldsymbol{\\alpha}_{0}) $.
            test (int): Check if the observation mismatches the predicted observation (AD test 1) or all
                        predicted symbols (AD test 2).
            delta (int): Maximum Hamming distance between the predicted observation and the actual observation (test 1).
            learning_rate (int): Pseudo-counts learning rate (use this parameter to boost the pseudo-counts updating).
        """

        # ----------------------------------------------------------------------
        # Check parameters
        # ----------------------------------------------------------------------

        # Check maximum number of distinct hidden state vectors against the maximum
        # number of equiprobable hypothesis in the state belief
        assert cs > ps > 0

        # Check pseudo-count parameter for the prior Dirichlet distribution
        assert alphas > 0

        # Check number of activated bits against the minimum distance between sparsely encoded observations
        assert az > bz > 0

        # Check maximum number of distinct encoded observations against the maximum
        # number of equiprobable hypothesis in the observation belief
        assert cz > pz > 0

        # Check pseudo-count parameter for the prior Dirichlet distribution
        assert alphaz > 0

        # Check the test type
        assert test == 1 or test == 2

        # Check the AD threshold for test 1
        assert delta > 0

        # Check the learning rate
        assert learning_rate > 0

        # ----------------------------------------------------------------------
        # Save the parameters
        # ----------------------------------------------------------------------
        self.cs = cs
        self.ps = ps
        self.alphas_0 = alphas * np.ones(cs, order='C', dtype=float)
        #js = np.unique(np.random.choice(self.cs, size=self.ps, replace=True))
        #self.alphas[js] = 1
        self.az = az
        self.bz = bz
        self.cz = cz
        self.dz = dz
        self.pz = pz
        self.alphaz_0 = alphaz * np.ones(cz, order='C', dtype=float)
        #jz = np.unique(np.random.choice(self.cz, size=self.pz, replace=True))
        #self.alphaz[jz] = 1
        self.test = test
        self.delta = delta
        self.learning_rate = learning_rate

        # ----------------------------------------------------------------------
        # Initialize members
        # ----------------------------------------------------------------------
        self.bs_n_n_1 = None
        self.bs_n_1_n = None
        self.bz_n_1_n = None
        self.wnn_layer0 = None
        self.wnn_layer1 = None
        self.wnn_layer2 = None
        self.z_n = None
        self.encoder = None

    def __del__(self):
        """
        Delete method. The garbage collector will hopefully work here.
        """

        # Clean up parameters
        self.cs = None
        self.ps = None
        self.alphas_0 = None
        self.az = None
        self.bz = None
        self.cz = None
        self.dz = None
        self.pz = None
        self.alphaz_0 = None
        self.test = None
        self.delta = None

        # Clean up members
        self.bs_n_1_n_2 = None
        self.bs_n_n_1 = None
        self.bs_n_1_n = None
        self.bz_n_1_n = None
        self.wnn_layer0 = None
        self.wnn_layer1 = None
        self.wnn_layer2 = None
        self.z_n_1 = None
        self.k_n_1 = None
        self.z_n = None
        self.k_n = None
        self.encoder = None

    def initialize(self):
        """
        Initialize the filter.
        """

        # ----------------------------------------------------------------------
        # Initialize the initial encoded observation $ \\check{\\bf z}_{0} $ by
        # activating up to $ a_{z} $ randomly chosen bits.
        # ----------------------------------------------------------------------
        z_0 = np.zeros(self.dz, order='C', dtype=bool)
        jz = np.unique(np.random.choice(self.dz, size=self.az, replace=True))
        z_0[jz] = True

        # ----------------------------------------------------------------------
        # Initialize layer 0: learn the initial encoded observation.
        # ----------------------------------------------------------------------
        self.wnn_layer0 = VGRAMNode.VGRAMNode(pattern_length=self.dz,
                                              min_mem_size=1, max_mem_size=2 ** 11,
                                              min_learn_dist=0, max_recall_dist=0)
        k_0 = self.wnn_layer0.learn(z_0, 1)

        # ----------------------------------------------------------------------
        # Draw the prior posterior $ \\boldsymbol{\pi}_{0|-1} $ from the Dirichlet
        # distribution $ Dir \\left( c_{s}; \\boldsymbol{\\alpha}_{0} \\right) $.
        # ----------------------------------------------------------------------
        pis_0 = np.random.dirichlet(self.alphas_0)

        # ----------------------------------------------------------------------
        # Draw up to $ p_{s} $ unique samples from the prior posterior $ \\boldsymbol{\pi}_{0|-1} $.
        # ----------------------------------------------------------------------
        ls_0 = np.unique(np.random.choice(self.cs, size=self.ps, replace=True, p=pis_0))

        # ----------------------------------------------------------------------
        # Build the prior belief $ {\\bf b}_{0|-1} $ using the samples.
        # ----------------------------------------------------------------------
        bs_0_0_1 = np.zeros(self.cs, order='C', dtype=bool)
        bs_0_0_1[ls_0] = True

        # ----------------------------------------------------------------------
        # Initialize the predicted state belief as the prior belief such that the
        # predicted equiprobable hypothesis are the same as in the prior belief.
        # ----------------------------------------------------------------------
        bs_0_1_0 = bs_0_0_1

        # ----------------------------------------------------------------------
        # Initialize layer 1: learn the transition from the prior to the predicted state belief.
        # ----------------------------------------------------------------------
        self.wnn_layer1 = VGRAMArray.VGRAMArray(output_dims=(1, self.cs), pattern_length=self.dz + self.cs,
                                                min_mem_size=1, max_mem_size=2 ** 11,
                                                #min_learn_dist=self.bz+self.ps/2, max_recall_dist=self.bz+self.ps/2,
                                                min_learn_dist=8, max_recall_dist=8,
                                                default_outputs=self.alphas_0)
        self.wnn_layer1.learn(np.concatenate((z_0, bs_0_0_1)), self.alphas_0+bs_0_1_0)

        # ----------------------------------------------------------------------
        # Initialize the predicted observation belief as a single hypotheses
        # corresponding to the same symbol as the initial observation.
        # ----------------------------------------------------------------------
        bz_0_1_0 = np.zeros(self.cz, order='C', dtype=bool)
        bz_0_1_0[k_0] = True

        # ----------------------------------------------------------------------
        # Initialize layer 2: learn the observation belief given the predicted state belief.
        # ----------------------------------------------------------------------
        self.wnn_layer2 = VGRAMArray.VGRAMArray(output_dims=(1, self.cz), pattern_length=self.cs,
                                                min_mem_size=1, max_mem_size=2 ** 11,
                                                #min_learn_dist=self.ps/2, max_recall_dist=self.ps/2,
                                                min_learn_dist=16, max_recall_dist=16,
                                                default_outputs=self.alphaz_0)
        self.wnn_layer2.learn(bs_0_1_0, self.alphaz_0+self.learning_rate*bz_0_1_0)

        # ----------------------------------------------------------------------
        # Initialize the previous observation as the initial one.
        # ----------------------------------------------------------------------
        self.z_n = z_0

        # ----------------------------------------------------------------------
        # Initialize previous state belief as the initial prior.
        # ----------------------------------------------------------------------
        self.bs_n_n_1 = bs_0_0_1

        # ----------------------------------------------------------------------
        # Initialize the scalar encoder.
        # ----------------------------------------------------------------------
        self.encoder = ScalarCodec.ScalarCodec(min_value=0, max_value=100,
                                               total_number_of_bits=self.dz, number_of_active_bits=self.az)

    def encode(self, y_n_1):
        """
        Encode the incoming observation $ \\check{\\bf y}_{n+1} $ at instant $ n+1 $.

        Parameters:
             y_n_1 (float[]): Observation vector at instant $ n + 1 $.
        Return:
            (bool[]): Sparsely encoded observation vector $ \\check{\\bf z}_{n+1} $ at instant $ n+1 $.
            (int): Index of the observation symbol in $ \\boldsymbol{\\Omega}_{z} $
        """

        # Encode the observation using the adaptive sparse encoder
        z_n_1 = self.encoder.encode(y_n_1)

        # Find the index of the corresponding input-output pair in memory
        [d_n_1, k_n_1] = self.wnn_layer0.find_closest_pattern(z_n_1)

        # Learn the new pattern if there is not a perfect match
        if d_n_1 != 0:
            k_n_1 = self.wnn_layer0.learn(z_n_1, 1)

        return z_n_1, k_n_1

    def decode(self, z_n_1_n):
        """
        Decode the predicted observation $ \\hat{\\bf y}_{n+1|n} $ at instant $ n+1 $.

        Parameters:
             z_n_1_n (bool[]): Sparsely encoded predicted observation vector at instant $ n + 1 $.

        Return:
            (float[]): Dense observation vector $ \\hat{\\bf y}_{n+1|n} $ at instant $ n+1 $.
        """

        # Encode the observation using the adaptive sparse encoder
        y_n_1_n = self.encoder.decode(z_n_1_n)

        return y_n_1_n

    def predict(self):
        """
        Predict the next encoded observation.

        Returns:
            (bool[]): Predicted encoded observation vector $ \\hat{\bf z}_{n+1|n} = \\boldsymbol{\\mu}^{(\\hat{k})} $.
            (int): Index of the predicted symbol.
            (float): Probability of the predicted symbol $ {\\tilde{\\mu}}^{(\\hat{k})}_{n+1|n} $.
        """

        # ----------------------------------------------------------------------
        # Algorithm 1: nP-GbF
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # Step 1: Retrieve hyper-parameters $ \\boldsymbol{\\alpha}_{n|n-1} $
        # indexed by $ \\check{\\bf z}_{n} $ and $ {\\bf b}_{n|n-1} $.
        # ----------------------------------------------------------------------
        alphas_n_n_1 = self.wnn_layer1.recall(np.concatenate((self.z_n, self.bs_n_n_1)))[0]

        # ----------------------------------------------------------------------
        # Step 2: Draw the predicted posterior $ \\boldsymbol{\\pi}_{n+1|n} $ from
        # the Dirichlet distribution $ Dir \\left( c_{s}; \\boldsymbol{\\alpha}_{n|n-1} \\right) $.
        # ----------------------------------------------------------------------
        pis_n_1_n = np.random.dirichlet(alphas_n_n_1)

        # ----------------------------------------------------------------------
        # Step 3: Draw up to $ p_{s} $ unique samples from the predicted posterior
        # $ \\boldsymbol{\\pi}_{n+1|n} $.
        # ----------------------------------------------------------------------
        ls_n_1_n = np.unique(np.random.choice(self.cs, size=self.ps, replace=True, p=pis_n_1_n))

        # ----------------------------------------------------------------------
        # Step 4: Build the predicted belief $ {\\bf b}_{n+1|n} $.
        # ----------------------------------------------------------------------
        self.bs_n_1_n = np.zeros(self.cs, order='C', dtype=bool)
        self.bs_n_1_n[ls_n_1_n] = True

        # ----------------------------------------------------------------------
        # Step 5: Retrieve the hyper-parameters $ \\tilde{\\boldsymbol{\\alpha}}_{n|n-1} $
        # indexed by $ {\\bf b}_{n+1|n} $.
        # ----------------------------------------------------------------------
        alphaz_n_n_1 = self.wnn_layer2.recall(self.bs_n_1_n)[0]

        # ----------------------------------------------------------------------
        # Step 6: Draw the predicted posterior $ \tilde{\\boldsymbol{\\pi}}_{n+1|n} $
        # from the Dirichlet distribution $ Dir \\left( c_{z}; \\tilde{\\boldsymbol{\\alpha}}_{n|n-1} \\right) $.
        # ----------------------------------------------------------------------
        piz_n_1_n = np.random.dirichlet(alphaz_n_n_1)

        # ----------------------------------------------------------------------
        # Step 7: Draw up to $ p_{z} $ unique samples from the predicted posterior
        # $ \\tilde{\\boldsymbol{\\pi}}_{n+1|n} $.
        # ----------------------------------------------------------------------
        lz_n_1_n = np.unique(np.random.choice(self.cz, size=self.pz, replace=True, p=piz_n_1_n))

        # ----------------------------------------------------------------------
        # Step 8: Build the predicted belief $ \\tilde{\\bf b}_{n+1|n} $.
        # ----------------------------------------------------------------------
        self.bz_n_1_n = np.zeros(self.cz, order='C', dtype=bool)
        self.bz_n_1_n[lz_n_1_n] = True

        # ----------------------------------------------------------------------
        # Step 9: Predict the next observation $ \\hat{\\bf z}_{n+1|n} $.
        # ----------------------------------------------------------------------
        k_n_1_n = np.argmax(piz_n_1_n)
        z_n_1_n = self.wnn_layer0.get_pattern_by_index(k_n_1_n)
        p_n_1_n = piz_n_1_n[k_n_1_n]

        return z_n_1_n, k_n_1_n, p_n_1_n

    def learn(self, z_n_1, k_n_1):
        """
        Update the hyperparameters upon the arrival of the next encoded observation $ \\check{\\bf z}_{n+1} $.

        Parameters:
            z_n_1 (bool[]): Sparsely encoded observation vector $ \\check{\bf z}_{n+1} $ at instant $ n+1 $.
            k_n_1 (int): Index of the sparse encoded observation at instant $ n+1 $.
        """

        # ----------------------------------------------------------------------
        # Algorithm 2: nP-MbL
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # Step 1: Check if there is a match.
        # ----------------------------------------------------------------------
        if self.bz_n_1_n[k_n_1] == 1:
            # ----------------------------------------------------------------------
            # Step 2: Retrieve the hyper-parameters $ \\boldsymbol{\\alpha}_{n|n-1} $
            # indexed by $ \\check{\\bf z}_{n} $ and $ {\\bf b}_{n|n-1} $.
            # ----------------------------------------------------------------------
            alphas_n_n_1 = self.wnn_layer1.recall(np.concatenate((self.z_n, self.bs_n_n_1)))[0]

            # ----------------------------------------------------------------------
            # Step 3: Update the hyper-parameters $ \\boldsymbol{\\alpha}_{n|n-1} $.
            # ----------------------------------------------------------------------
            alphas_n_1_n = alphas_n_n_1 + self.bs_n_1_n

            # ----------------------------------------------------------------------
            # Step 4: Retrieve the hyper-parameters $ \\tilde{\\boldsymbol{\\alpha}}_{n|n-1} $
            # indexed by $ {\\bf b}_{n+1|n} $.
            # ----------------------------------------------------------------------
            alphaz_n_n_1 = self.wnn_layer2.recall(self.bs_n_1_n)[0]

            # ----------------------------------------------------------------------
            # Step 5: Update the hyper-parameters $ \\tilde{\\boldsymbol{\\alpha}}_{n|n-1} $.
            # ----------------------------------------------------------------------
            alphaz_n_1_n = alphaz_n_n_1 + self.bz_n_1_n
            alphaz_n_1_n[k_n_1] += self.learning_rate
        else:  # Step 6...
            # ----------------------------------------------------------------------
            # Draw a new predicted posterior $ \\boldsymbol{\\pi}_{n+1|n} $ from the Dirichlet
            # distribution $ Dir \\left( c_{s}; \\boldsymbol{\\alpha}_{0} \\right) $.
            # ----------------------------------------------------------------------
            pis_n_1_n = np.random.dirichlet(self.alphas_0)

            # ----------------------------------------------------------------------
            # Draw up to $ p_{s} $ unique samples from the prior posterior $ \\boldsymbol{\\pi}_{n+1|n} $.
            # ----------------------------------------------------------------------
            ls_n_1_n = np.unique(np.random.choice(self.cs, size=self.ps, replace=True, p=pis_n_1_n))

            # ----------------------------------------------------------------------
            # Build the prior belief $ {\\bf b}_{n+1|n} $.
            # ----------------------------------------------------------------------
            self.bs_n_1_n = np.zeros(self.cs, order='C', dtype=bool)
            self.bs_n_1_n[ls_n_1_n] = True

            # ----------------------------------------------------------------------
            # Steps 7,8: Initialize the hyper-parameters $ \\boldsymbol{\\alpha}_{n+1|n} $.
            # ----------------------------------------------------------------------
            alphas_n_1_n = self.alphas_0 + self.bs_n_1_n

            # ----------------------------------------------------------------------
            # Steps 9,10: Initialize the hyper-parameters $ \\tilde{\\boldsymbol{\\alpha}}_{n+1|n} $.
            # ----------------------------------------------------------------------
            alphaz_n_1_n = self.alphaz_0
            alphaz_n_1_n[k_n_1] += self.learning_rate
        # Step 11...

        # ----------------------------------------------------------------------
        # Steps 12,13: Store the updated hyper-parameters.
        # ----------------------------------------------------------------------
        self.wnn_layer1.learn(np.concatenate((self.z_n, self.bs_n_n_1)), alphas_n_1_n)
        self.wnn_layer2.learn(self.bs_n_1_n, alphaz_n_1_n)

        # Update the previous observations
        self.z_n = z_n_1

        # Update the previous beliefs
        self.bs_n_n_1 = self.bs_n_1_n

    def handle(self, y_n_1):
        """
        Handle the current encoded observation using the non-parametric, CL anomaly detector (nP-CLAD).

        Parameters:
            y_n_1 (float[]): Observation vector at instant $ n + 1 $.

        Returns:
            (bool): True if the current observation is an anomaly, false otherwise.
            (float): Anomaly score indicating how likely the current observation is an anomaly.
            (float[]): Predicted dense observation vector $ \\hat{\bf y}_{n+1|n} = \\boldsymbol{\\Omega}_{y} $.
        """

        # ----------------------------------------------------------------------
        # Algorithm 3: nP-CLAD
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # Step 1: Encode observation.
        # ----------------------------------------------------------------------
        z_n_1, k_n_1 = self.encode(y_n_1)

        # ----------------------------------------------------------------------
        # Step 2: Predict next observation.
        # ----------------------------------------------------------------------
        z_n_1_n, k_n_1_n, p_n_1_n = self.predict()

        # ----------------------------------------------------------------------
        # Step 3,4: Detect the anomaly and compute the corresponding score.
        # ----------------------------------------------------------------------
        if self.test == 1:
            # Perform test 1: check if the predicted observation mismatches the encoded observation
            anomaly_n_1 = z_n_1_n is not None and np.count_nonzero(z_n_1_n != z_n_1) > self.delta
            score_n_1 = 1 - p_n_1_n
        elif self.test == 2:
            # Perform test 2: check if the encoded observation (index) mismatches all predicted hypothesis
            anomaly_n_1 = self.bz_n_1_n[k_n_1] == 0
            score_n_1 = 1 - np.count_nonzero(self.bz_n_1_n)/self.cs
        else:
            raise Exception("Invalid AD test type.")

        # ----------------------------------------------------------------------
        # Step 5: Learn the non-parametric model.
        # ----------------------------------------------------------------------
        self.learn(z_n_1, k_n_1)

        # ----------------------------------------------------------------------
        # Step 6: Decode the predicted observation as $ \\hat{\\bf y}_{n+1|n} $.
        # ----------------------------------------------------------------------
        y_n_1_n = self.decode(z_n_1_n)

        return anomaly_n_1, score_n_1, y_n_1_n

    def memory_size(self):
        """
        Network memory size.

        Return:
            (float): layer 0 memory size (MB)
            (float): layer 1 memory size (MB)
            (float): layer 2 memory size (MB)
        """
        layer0_mem_size_kilobytes, _, _ = self.wnn_layer0.memory_stats()
        layer1_mem_size_megabytes, _, _ = self.wnn_layer1.memory_stats()
        layer2_mem_size_megabytes, _, _ = self.wnn_layer2.memory_stats()
        layer0_mem_size_megabytes = layer0_mem_size_kilobytes / 1024
        return layer0_mem_size_megabytes, layer1_mem_size_megabytes, layer2_mem_size_megabytes

    def debug(self):
        """
        Debug network.
        """
        self.wnn_layer0.debug()
        self.wnn_layer1.debug()
        self.wnn_layer2.debug()


class TestNPCLAD(unittest.TestCase):
    """
    Extends unittest.TestCase class to implement unit tests for the NPCLAD class.
    """

    # Model parameters
    cs = 2 ** 11
    ps = 2 ** 5
    alphas = 2 ** -12
    az = 2 ** 6
    bz = 2 ** 3
    cz = 2 ** 11
    dz = 2 ** 9
    pz = 2 ** 4
    alphaz = 2 ** -12
    test = 1
    delta = 8
    learning_rate = 8
    detector = None
    test_statistics = None

    # Time-series parameters
    time_series_length = 100
    time_indexes = range(0, time_series_length)

    @classmethod
    def setUpClass(cls):
        """
        Set up method: configure parameters and create a VGRAM node.
        """
        cls.detector = NPCLAD(cs=cls.cs, ps=cls.ps, alphas=cls.alphas,
                              az=cls.az, bz=cls.bz, cz=cls.cz, dz=cls.dz, pz=cls.pz, alphaz=cls.alphaz,
                              test=cls.test, delta=cls.delta, learning_rate=cls.learning_rate)
        cls.test_statistics = {}

    @classmethod
    def tearDownClass(cls):
        """
        Tear down method: print test statistics.
        """
        print('Elapsed time to detect {0} patterns: {1:.2e} seconds'.format(cls.time_series_length,
                                                                            cls.test_statistics["elapsed_detect_time"]))
        print('Average learn time: {:.1} milliseconds'.format(cls.test_statistics["average_detect_time"]))

        time_series_name = cls.test_statistics["time_series_name"]
        time_series = cls.test_statistics["time_series"]
        ground_truth = cls.test_statistics["ground_truth"]
        anomalies = cls.test_statistics["anomalies"]
        scores = cls.test_statistics["scores"]
        predictions = cls.test_statistics["predictions"]
        layer0_memory_stats = cls.test_statistics["layer0_memory_stats"]
        layer1_memory_stats = cls.test_statistics["layer1_memory_stats"]
        layer2_memory_stats = cls.test_statistics["layer2_memory_stats"]

        plt.subplots()
        plt.subplot(511)
        plt.plot(time_series, 'bo--',  label='Time-series')
        plt.plot(predictions, 'm.--',  label='Predictions')
        plt.xticks(np.arange(0, cls.time_series_length, step=cls.time_series_length/20))
        plt.grid(axis='x', color='0.95')
        plt.title(time_series_name)
        plt.legend(loc="upper right")
        plt.subplot(512)
        plt.yticks([1.0, 0.0], ["True", "False"])
        cmap = clrs.ListedColormap(['green', 'red'])
        plt.scatter(x=cls.time_indexes, y=ground_truth, c=ground_truth.astype(float), marker='d', cmap=cmap)
        plt.xticks(np.arange(0, cls.time_series_length, step=cls.time_series_length/20))
        plt.grid(axis='x', color='0.95')
        plt.title('Ground-truth')
        plt.subplot(513)
        plt.yticks([1.0, 0.0], ["True", "False"])
        cmap = clrs.ListedColormap(['green', 'red'])
        plt.scatter(x=cls.time_indexes, y=anomalies, c=anomalies.astype(float), marker='d', cmap=cmap)
        plt.xticks(np.arange(0, cls.time_series_length, step=cls.time_series_length/20))
        plt.grid(axis='x', color='0.95')
        plt.title('Anomaly indicator')
        plt.subplot(514)
        plt.plot(100*scores, 'b-')
        plt.xticks(np.arange(0, cls.time_series_length, step=cls.time_series_length/20))
        plt.grid(axis='x', color='0.95')
        plt.title('Anomaly score (%)')
        plt.subplot(515)
        plt.plot(layer0_memory_stats, 'r-',  label='Layer 0')
        plt.plot(layer1_memory_stats, 'g-',  label='Layer 1')
        plt.plot(layer2_memory_stats, 'b-',  label='Layer 2')
        plt.xticks(np.arange(0, cls.time_series_length, step=cls.time_series_length/20))
        plt.grid(axis='x', color='0.95')
        plt.title('Memory usage (MB)')
        plt.legend(loc="upper right")
        plt.show()

        cls.detector = None
        cls.test_statistics = None

    def test_0_cte_time_series_with_spike_anomalies(self):
        """
        Test case 0: Constant valued time-series with abnormal spikes.
        """

        # Time-series parameters
        time_series_cte_value = 10
        number_of_spikes = 3
        spike_value = 100

        # Build the time-series
        time_series = time_series_cte_value * np.ones(self.time_series_length, order='C', dtype=float)
        spike_locations = np.unique(np.random.choice(self.time_series_length, number_of_spikes, replace=True))
        time_series[spike_locations] = spike_value

        # Build the ground_truth
        ground_truth = np.zeros(self.time_series_length, order='C', dtype=bool)
        ground_truth[spike_locations] = True

        # Output vectors
        anomalies = np.zeros(self.time_series_length, order='C', dtype=bool)
        scores = np.zeros(self.time_series_length, order='C', dtype=float)
        predictions = np.zeros(self.time_series_length, order='C', dtype=float)

        # Memory statistics
        layer0_memory_stats = np.zeros(self.time_series_length, order='C', dtype=float)
        layer1_memory_stats = np.zeros(self.time_series_length, order='C', dtype=float)
        layer2_memory_stats = np.zeros(self.time_series_length, order='C', dtype=float)

        # Initialize the detector
        self.detector.initialize()

        # Run the detector
        elapsed_time = 0
        for n in self.time_indexes:
            value = time_series[n]
            t = time.time()
            anomaly, score, predicted_value = self.detector.handle(value)
            elapsed_time += time.time() - t
            anomalies[n] = anomaly
            scores[n] = score
            predictions[n] = predicted_value
            layer0_memory_stats[n], layer1_memory_stats[n], layer2_memory_stats[n] = self.detector.memory_size()
            self.detector.debug()

        # Collect the statistics
        self.test_statistics["elapsed_detect_time"] = elapsed_time
        self.test_statistics["average_detect_time"] = 100 * elapsed_time / self.time_series_length
        self.test_statistics["time_series_name"] = 'Constant-valued time-series with random spikes'
        self.test_statistics["time_series"] = time_series
        self.test_statistics["ground_truth"] = ground_truth
        self.test_statistics["anomalies"] = anomalies
        self.test_statistics["scores"] = scores
        self.test_statistics["predictions"] = predictions
        self.test_statistics["layer0_memory_stats"] = layer0_memory_stats
        self.test_statistics["layer1_memory_stats"] = layer1_memory_stats
        self.test_statistics["layer2_memory_stats"] = layer2_memory_stats

    def test_1_sin_time_series_with_spike_anomalies(self):
        """
        Test case 1: Sinusoidal time-series with abnormal spikes.
        """

        # Time-series parameters
        time_series_amplitude = 15
        time_series_offset = 15
        time_series_frequency = 2*np.pi/10
        time_series_phase = 0
        number_of_spikes = 3
        spike_value = 100

        # Build the time-series
        time_series = time_series_offset + time_series_amplitude * \
            np.sin(np.array(self.time_indexes, order='C', dtype=float) * time_series_frequency + time_series_phase)
        spike_locations = np.unique(np.random.choice(self.time_series_length, number_of_spikes, replace=True))
        time_series[spike_locations] = spike_value

        # Build the ground_truth
        ground_truth = np.zeros(self.time_series_length, order='C', dtype=bool)
        ground_truth[spike_locations] = True

        # Output vectors
        anomalies = np.zeros(self.time_series_length, order='C', dtype=bool)
        scores = np.zeros(self.time_series_length, order='C', dtype=float)
        predictions = np.zeros(self.time_series_length, order='C', dtype=float)

        # Memory statistics
        layer0_memory_stats = np.zeros(self.time_series_length, order='C', dtype=float)
        layer1_memory_stats = np.zeros(self.time_series_length, order='C', dtype=float)
        layer2_memory_stats = np.zeros(self.time_series_length, order='C', dtype=float)

        # Initialize the detector
        self.detector.initialize()

        # Run the detector
        elapsed_time = 0
        for n in self.time_indexes:
            value = time_series[n]
            t = time.time()
            anomaly, score, predicted_value = self.detector.handle(value)
            elapsed_time += time.time() - t
            anomalies[n] = anomaly
            scores[n] = score
            predictions[n] = predicted_value
            layer0_memory_stats[n], layer1_memory_stats[n], layer2_memory_stats[n] = self.detector.memory_size()
            self.detector.debug()

        # Collect the statistics
        self.test_statistics["elapsed_detect_time"] = elapsed_time
        self.test_statistics["average_detect_time"] = 100 * elapsed_time / self.time_series_length
        self.test_statistics["time_series_name"] = 'Sinusoidal time-series with random spikes'
        self.test_statistics["time_series"] = time_series
        self.test_statistics["ground_truth"] = ground_truth
        self.test_statistics["anomalies"] = anomalies
        self.test_statistics["scores"] = scores
        self.test_statistics["predictions"] = predictions
        self.test_statistics["layer0_memory_stats"] = layer0_memory_stats
        self.test_statistics["layer1_memory_stats"] = layer1_memory_stats
        self.test_statistics["layer2_memory_stats"] = layer2_memory_stats


if __name__ == '__main__':
    unittest.main()

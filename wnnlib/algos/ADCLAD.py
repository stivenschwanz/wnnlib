from numpy import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import unittest
import time
from wnnlib.vgram import VGRAMArray, VGRAMNode
from wnnlib.codecs import FixedScalarCodec, AdaptiveScalarCodec
from wnnlib.utils.BitUtils import BitUtils
import gc
from scipy import signal


class ADCLAD:
    """
    This class implements a continuous learning (CL), anomaly detector (AD) for streamed sequences of observations
    using a non-parametric Bayesian procedure to build a suitable model of the underlying stochastic process emitting
    the observations.

    Inefficient, slightly pretty, hopefully correct Pythonic (yuck) implementation of the AD-CLAD method.
    """

    def __init__(self, cs, ps, alphas,
                 az, bz, cz, dz, pz, alphaz,
                 test=1, delta=0, tau=0.75,
                 learning_rate=1, sub_seq_len=4):
        """
        Initialize the AD-CLAD detector using the nine hyperparameters.

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
            tau (float): Anomaly score threshold (test 2).
            learning_rate (int): Pseudo-counts learning rate (use this parameter to boost the pseudo-counts updating).
            sub_seq_len (int): Length of the sub-sequences employed to build the hidden state symbols.
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
        assert delta >= 0

        # Check the AD threshold for test 2
        assert 1 >= tau >= 0

        # Check the learning rate
        assert learning_rate > 0

        # Check the sub_seq_len
        assert sub_seq_len > 1

        # ----------------------------------------------------------------------
        # Save the parameters
        # ----------------------------------------------------------------------
        self.cs = cs
        self.ps = ps
        self.alphas_0 = alphas * np.ones(cs, order='C', dtype=float)
        self.az = az
        self.bz = bz
        self.cz = cz
        self.dz = dz
        self.pz = pz
        self.alphaz_0 = alphaz * np.ones(cz, order='C', dtype=float)
        self.test = test
        self.delta = delta
        self.tau = tau
        self.learning_rate = learning_rate
        self.sub_seq_len = sub_seq_len

        # ----------------------------------------------------------------------
        # Initialize members
        # ----------------------------------------------------------------------
        self.wnn_layer0 = None
        self.wnn_layer1 = None
        self.wnn_layer2 = None
        self.bs_n_n_1 = None
        self.bs_n_1_n = None
        self.piz_n_1_n = None
        self.pis_n_1_n = None
        self.z_n = None
        self.k_n = None
        self.codec = None
        self.z_counter = 0
        self.attention = None
        self.sliding_window = None
        self.sub_seq_counter = None
        self.sub_seqs = None
        self.curr_sub_seq = None
        self.j_n_1 = None
        self.n_1 = None
        self.min_time_series_value = None
        self.max_time_series_value = None
        self.time_series_length = None

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
        self.tau = None
        self.learning_rate = None
        self.sub_seq_len = None

        # Clean up members
        self.wnn_layer0 = None
        self.wnn_layer1 = None
        self.wnn_layer2 = None
        self.bs_n_n_1 = None
        self.bs_n_1_n = None
        self.piz_n_1_n = None
        self.pis_n_1_n = None
        self.z_n = None
        self.k_n = None
        self.codec = None
        self.z_counter = 0
        self.attention = None
        self.sliding_window = None
        self.sub_seq_counter = None
        self.sub_seqs = None
        self.curr_sub_seq = None
        self.j_n_1 = None
        self.n_1 = None
        self.min_time_series_value = None
        self.max_time_series_value = None
        self.time_series_length = None

        gc.collect()
        # print(gc.get_stats())

    def encode(self, y_n_1):
        """
        Encode the incoming observation $ \\check{\\bf y}_{n+1} $ at instant $ n+1 $.

        Parameters:
             y_n_1 (float[]): Observation vector at instant $ n + 1 $.
        Return:
            (bool[]): Sparsely encoded observation vector $ \\check{\\bf z}_{n+1} $ at instant $ n+1 $.
            (int): Index of the observation symbol in $ \\boldsymbol{\\Omega}_{z} $.
        """

        # Encode the observation using the adaptive sparse encoder
        z_n_1 = self.codec.encode(y_n_1)

        # Find the index of the corresponding input-output pair in memory
        _, _, k_n_1 = self.wnn_layer0.recall(z_n_1)

        # Check if there is a match
        if k_n_1 is None:
            # Learn the new pattern and retrieve the index of the corresponding input-output pair in memory
            k_n_1 = self.wnn_layer0.learn(z_n_1, self.z_counter)

            # Increment the global counter for statistics purposes
            self.z_counter += 1

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
        y_n_1_n = self.codec.decode(z_n_1_n)

        return y_n_1_n

    def initialize(self, min_time_series_value=None, max_time_series_value=None, time_series_length=None):
        """
        Initialize the filter.
        """

        # ----------------------------------------------------------------------
        # Initialize time step counter.
        # ----------------------------------------------------------------------
        self.n_1 = 0
        self.min_time_series_value = min_time_series_value
        self.max_time_series_value = max_time_series_value
        self.time_series_length = time_series_length

        # ----------------------------------------------------------------------
        # Initialize codec: a flexible scalar encoder / decoder mapping real
        # number into sparse representations with $ d_{z} $ bit and up to $ a_{z} $
        # active bits.
        # ----------------------------------------------------------------------
        if min_time_series_value is None or max_time_series_value is None:
            self.codec = AdaptiveScalarCodec.AdaptiveScalarCodec(number_of_active_bits=self.az,
                                                                 total_number_of_bits=self.dz,
                                                                 max_window_size=100)
        else:
            self.codec = FixedScalarCodec.FixedScalarCodec(min_value=min_time_series_value,
                                                           max_value=max_time_series_value,
                                                           number_of_active_bits=self.az,
                                                           total_number_of_bits=self.dz)

        # ----------------------------------------------------------------------
        # Initialize layer 0: a single-node layer to store up to $ c_{z} $ distinct
        # symbols (encoded observations with $ d_{z} $ bits) at least $ b_{z} $ bits
        # apart from each other. The default output value must be zero.
        # ----------------------------------------------------------------------
        self.wnn_layer0 = VGRAMNode.VGRAMNode(pattern_length=self.dz,
                                              min_mem_size=self.cz - 1, max_mem_size=self.cz,
                                              min_learn_dist=self.bz, max_recall_dist=self.az // 4,
                                              default_output=None, type_output=int)

        # ----------------------------------------------------------------------
        # Initialize the previous observation.
        # ----------------------------------------------------------------------
        if min_time_series_value is None or max_time_series_value is None:
            y_0 = 0
        else:
            y_0 = (min_time_series_value + max_time_series_value) / 2

        self.z_n, self.k_n = self.encode(y_0)

        # ----------------------------------------------------------------------
        # Initialize the previous hidden state.
        # ----------------------------------------------------------------------
        self.sub_seqs = {}
        self.curr_sub_seq = (self.k_n,) * self.sub_seq_len
        self.sub_seq_counter = 0
        self.j_n_1 = self.sub_seq_counter
        self.sub_seqs[self.curr_sub_seq] = self.j_n_1

        # ----------------------------------------------------------------------
        # Initialize layer 1: learn the transition from the prior to the predicted state belief.
        # ----------------------------------------------------------------------
        self.wnn_layer1 = VGRAMArray.VGRAMArray(output_dims=(1, self.cs), pattern_length=self.dz + self.cs + self.ps,
                                                min_mem_size=2 ** 12 - 1, max_mem_size=2 ** 12,
                                                min_learn_dist=(self.bz + self.ps) // 2,
                                                max_recall_dist=(self.bz + self.ps) // 2,
                                                default_outputs=self.alphas_0, type_outputs=np.float64)

        # ----------------------------------------------------------------------
        # Initialize layer 2: learn the observation belief given the predicted state belief.
        # ----------------------------------------------------------------------
        self.wnn_layer2 = VGRAMArray.VGRAMArray(output_dims=(1, self.cz), pattern_length=self.cs + self.ps,
                                                min_mem_size=2 ** 12 - 1, max_mem_size=2 ** 12,
                                                min_learn_dist=self.ps // 2, max_recall_dist=self.ps // 2,
                                                default_outputs=self.alphaz_0, type_outputs=np.float64)

        # ----------------------------------------------------------------------
        # Initialize previous state belief as the initial prior.
        # ----------------------------------------------------------------------
        as_0_0_1 = np.copy(self.alphas_0)
        as_0_0_1[self.j_n_1] += self.learning_rate
        bs_0_0_1, _, _ = BitUtils.belief(self.cs, self.ps, as_0_0_1)
        self.bs_n_n_1 = np.copy(bs_0_0_1)

    def predict(self):
        """
        Predict the next encoded observation.

        Returns:
            (bool[]): Predicted encoded observation vector $ \\hat{\bf z}_{n+1|n} = \\boldsymbol{\\mu}^{(\\hat{k})} $.
            (int): Index of the predicted observation symbol.
            (float): Probability of the predicted observation symbol $ {\\tilde{\\mu}}^{(\\hat{k})}_{n+1|n} $.
            (int): Index of the predicted state symbol.
            (float): Probability of the predicted state symbol $ {\\mu}^{(\\hat{j})}_{n+1|n} $.
        """

        # ----------------------------------------------------------------------
        # Algorithm 1: nP-GbF
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # Retrieve hyper-parameters $ \\boldsymbol{\\alpha}_{n|n-1} $
        # indexed by $ \\check{\\bf z}_{n} $ and $ {\\bf b}_{n|n-1} $.
        # ----------------------------------------------------------------------
        alphas_n_n_1 = self.wnn_layer1.recall(np.concatenate((self.z_n, self.bs_n_n_1)))[0]

        # ----------------------------------------------------------------------
        # Build the predicted belief $ {\\bf b}_{n+1|n} $ from the retrieved
        # hyper-parameters $ \\boldsymbol{\\alpha}_{n|n-1} $.
        # ----------------------------------------------------------------------
        self.bs_n_1_n, _, self.pis_n_1_n = BitUtils.belief(self.cs, self.ps, alphas_n_n_1)

        # ----------------------------------------------------------------------
        # Predict the next state $ \\hat{\\bf s}_{n+1|n} $ for debug purposes.
        # ----------------------------------------------------------------------
        j_n_1_n = np.argmax(self.pis_n_1_n)
        ps_n_1_n = self.pis_n_1_n[j_n_1_n]

        # ----------------------------------------------------------------------
        # Retrieve the hyper-parameters $ \\tilde{\\boldsymbol{\\alpha}}_{n|n-1} $
        # indexed by $ {\\bf b}_{n+1|n} $.
        # ----------------------------------------------------------------------
        alphaz_n_n_1 = self.wnn_layer2.recall(self.bs_n_1_n)[0]

        # ----------------------------------------------------------------------
        # Draw the predicted posterior $ \\tilde{\\bf p}_{n+1|n} $ from the retrieved
        # hyper-parameters $ \\tilde{\\boldsymbol{\\alpha}}_{n|n-1} $.
        # ----------------------------------------------------------------------
        _, _, self.piz_n_1_n = BitUtils.belief(self.cz, self.pz, alphaz_n_n_1)

        # ----------------------------------------------------------------------
        # Predict the next observation $ \\hat{\\bf z}_{n+1|n} $.
        # ----------------------------------------------------------------------
        k_n_1_n = np.argmax(self.piz_n_1_n)
        z_n_1_n = self.wnn_layer0.get_pattern_by_output(k_n_1_n)
        pz_n_1_n = self.piz_n_1_n[k_n_1_n]

        return z_n_1_n, k_n_1_n, pz_n_1_n, j_n_1_n, ps_n_1_n

    def learn(self, z_n_1, k_n_1, j_n_1):
        """
        Update the hyperparameters upon the arrival of the next encoded observation $ \\check{\\bf z}_{n+1} $.

        Parameters:
            z_n_1 (bool[]): Sparsely encoded observation vector $ \\check{\bf z}_{n+1} $ at instant $ n+1 $.
            k_n_1 (int): Index of the sparse encoded observation at instant $ n+1 $.
            j_n_1 (int): Index of the sparse encoded state at instant $ n+1 $.
        """

        # ----------------------------------------------------------------------
        # Algorithm 2: nP-MbL
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # Retrieve the hyper-parameters $ \\boldsymbol{\\alpha}_{n|n-1} $
        # indexed by $ \\check{\\bf z}_{n} $ and $ {\\bf b}_{n|n-1} $.
        # ----------------------------------------------------------------------
        alphas_n_n_1 = self.wnn_layer1.recall(np.concatenate((self.z_n, self.bs_n_n_1)))[0]

        # ----------------------------------------------------------------------
        # Update the hyper-parameters $ \\boldsymbol{\\alpha}_{n+1|n} $.
        # ----------------------------------------------------------------------
        alphas_n_1_n = np.copy(alphas_n_n_1)
        alphas_n_1_n[j_n_1] += self.learning_rate

        # ----------------------------------------------------------------------
        # Store the updated hyper-parameters.
        # ----------------------------------------------------------------------
        self.wnn_layer1.learn(np.concatenate((self.z_n, self.bs_n_n_1)), alphas_n_1_n)

        # ----------------------------------------------------------------------
        # Rebuild the predicted belief $ {\\bf b}_{n+1|n} $ from the updated
        # hyper-parameters $ \\boldsymbol{\\alpha}_{n+1|n} $.
        # ----------------------------------------------------------------------
        self.bs_n_1_n, _, _ = BitUtils.belief(self.cs, self.ps, alphas_n_1_n)

        # ----------------------------------------------------------------------
        # Retrieve the hyper-parameters $ \\tilde{\\boldsymbol{\\alpha}}_{n|n-1} $
        # indexed by $ {\\bf b}_{n+1|n} $.
        # ----------------------------------------------------------------------
        alphaz_n_n_1 = self.wnn_layer2.recall(self.bs_n_1_n)[0]

        # ----------------------------------------------------------------------
        # Update the hyper-parameters $ \\tilde{\\boldsymbol{\\alpha}}_{n+1|n} $.
        # ----------------------------------------------------------------------
        alphaz_n_1_n = np.copy(alphaz_n_n_1)
        alphaz_n_1_n[k_n_1] += self.learning_rate
        alphaz_n_1_n[k_n_1] = min(alphaz_n_1_n[k_n_1], 16)

        # ----------------------------------------------------------------------
        # Store the updated hyper-parameters.
        # ----------------------------------------------------------------------
        self.wnn_layer2.learn(self.bs_n_1_n, alphaz_n_1_n)

        # ----------------------------------------------------------------------
        # Update the previous observations
        # ----------------------------------------------------------------------
        self.z_n = np.copy(z_n_1)
        self.k_n = k_n_1

        # ----------------------------------------------------------------------
        # Update the previous belief
        # ----------------------------------------------------------------------
        self.bs_n_n_1 = np.copy(self.bs_n_1_n)

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

        # Increment the time instant
        self.n_1 += 1

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
        z_n_1_n, k_n_1_n, _, j_n_1_n, _ = self.predict()

        # ----------------------------------------------------------------------
        # Step 3: Compute the anomaly score.
        # ----------------------------------------------------------------------
        if z_n_1_n is not None:
            score_n_1 = 1.0 - self.piz_n_1_n[k_n_1]
        else:
            score_n_1 = 0.0

        # ----------------------------------------------------------------------
        # Step 4: Detect the anomaly and compute the corresponding score.
        # ----------------------------------------------------------------------
        if z_n_1_n is not None:
            # Check if the encoded observation (index) mismatches all predicted hypothesis
            anomaly_n_1 = score_n_1 > self.tau
        else:
            anomaly_n_1 = False

        # ----------------------------------------------------------------------
        # Update the current sub-sequence (shift it to the left and add the new index)
        # ----------------------------------------------------------------------
        self.curr_sub_seq = (k_n_1,) + self.curr_sub_seq[:-1]
        if self.curr_sub_seq in self.sub_seqs:
            self.j_n_1 = self.sub_seqs[self.curr_sub_seq]
        else:
            self.sub_seq_counter += self.ps
            if self.sub_seq_counter >= self.cs:
                self.sub_seq_counter += 1
                self.sub_seq_counter %= self.cs
            self.j_n_1 = self.sub_seq_counter
            self.sub_seqs[self.curr_sub_seq] = self.j_n_1

        # ----------------------------------------------------------------------
        # Step 5: Learn the non-parametric model.
        # ----------------------------------------------------------------------
        self.learn(z_n_1, k_n_1, self.j_n_1)

        # ----------------------------------------------------------------------
        # Step 6: Decode the predicted observation as $ \\hat{\\bf y}_{n+1|n} $.
        # ----------------------------------------------------------------------
        y_n_1_n = self.decode(z_n_1_n)

        return anomaly_n_1, score_n_1, y_n_1_n, k_n_1_n, k_n_1, j_n_1_n

    def memory_size(self):
        """
        Network memory size.

        Return:
            (float): layer 0 memory size (MB).
            (float): layer 1 memory size (MB).
            (float): layer 2 memory size (MB).
        """
        layer0_mem_size_kilobytes, _, _ = self.wnn_layer0.memory_stats()
        layer0_mem_size_megabytes = layer0_mem_size_kilobytes / 1024
        layer1_mem_size_megabytes, _, _ = self.wnn_layer1.memory_stats()
        layer2_mem_size_megabytes, _, _ = self.wnn_layer2.memory_stats()
        return layer0_mem_size_megabytes, layer1_mem_size_megabytes, layer2_mem_size_megabytes

    def debug(self):
        """
        Debug network.
        """
        # self.wnn_layer0.debug()
        # self.wnn_layer1.debug()
        # self.wnn_layer2.debug()
        # gc.collect()
        # print(gc.get_stats())


class TestNPCLAD(unittest.TestCase):
    """
    Extends unittest.TestCase class to implement unit tests for the NPCLAD class.
    """

    # Model parameters
    cs = 2 ** 10
    ps = 2 ** 3
    alphas = 2 ** -12
    az = 64
    bz = 2 ** 2
    cz = 2 ** 10
    dz = 2 ** 10
    pz = 2 ** 3
    alphaz = 2 ** -12
    test = 2
    min_overlap = 52  # Minimum overlap between the predicted observation and the encoded observation
    delta = 2 * az - 2 * min_overlap
    tau = 0.75
    learning_rate = 2 ** 1
    sub_seq_len = 2 ** 3

    detector = None
    test_statistics = None
    plot_pretty_graphs = True
    plot_statistics = False

    # Time-series parameters
    time_series_length = 1001
    time_indexes = range(0, time_series_length)
    anomaly_indexes = range(time_series_length // 2, time_series_length)
    sampling_frequency = 10  # [Hz]
    sampling_period = 1 / sampling_frequency  # [secs]
    time_instants = np.array(time_indexes, order='C', dtype=float) * sampling_period  # [secs]

    def setUp(self):
        """
        Set up method: configure parameters and create a VGRAM node.
        """
        self.detector = NPCLAD(cs=self.cs, ps=self.ps, alphas=self.alphas,
                               az=self.az, bz=self.bz, cz=self.cz, dz=self.dz, pz=self.pz, alphaz=self.alphaz,
                               test=self.test, delta=self.delta, tau=self.tau,
                               learning_rate=self.learning_rate, sub_seq_len=self.sub_seq_len)
        self.test_statistics = {}

    def tearDown(self):
        """
        Tear down method: print test statistics.
        """
        print('Elapsed time to detect {0} patterns: {1:.2e} seconds'.format(self.time_series_length,
                                                                            self.test_statistics[
                                                                                "elapsed_detect_time"]))
        print('Mean detection time: {0:.2f} milliseconds'.format(self.test_statistics["mean_detect_time"]*1000.0))
        print('Max detection time: {0:.2f} milliseconds'.format(self.test_statistics["max_detect_time"]*1000.0))
        print('Acc network memory: {0:.2f} kilobytes'.format(self.test_statistics["total_memory_stats"]))

        time_series_name = self.test_statistics["time_series_name"]
        time_series = self.test_statistics["time_series"]
        ground_truth = self.test_statistics["ground_truth"]
        anomalies = self.test_statistics["anomalies"]
        scores = self.test_statistics["scores"]
        predictions = self.test_statistics["predictions"]
        predicted_observation_symbols = self.test_statistics["predicted_observation_symbols"]
        observation_symbols = self.test_statistics["observation_symbols"]
        predicted_state_symbols = self.test_statistics["predicted_state_symbols"]
        layer0_memory_stats = self.test_statistics["layer0_memory_stats"]
        layer1_memory_stats = self.test_statistics["layer1_memory_stats"]
        layer2_memory_stats = self.test_statistics["layer2_memory_stats"]

        if self.plot_pretty_graphs:
            fig = plt.figure(figsize=(16, 6))
            gs = fig.add_gridspec(2, hspace=0, height_ratios=[0.65, 0.35])
            axs = gs.subplots(sharex=True, sharey=False)

            axs[0].set_frame_on(False)
            axs[0].grid(True)
            axs[0].set_xlim(0, 1000)
            axs[0].set_ylim(0, 100)
            axs[0].axvspan(0, self.time_series_length / 2,
                           color='gray', alpha=0.1, label='Learning-only: $500$ points without anomalies')
            axs[0].axvspan(self.time_series_length / 2, self.time_series_length,
                           color='green', alpha=0.1, label='Evaluation: $500$ points with abnormal spikes')
            anomaly_time_indexes = np.array(self.time_indexes)[ground_truth]
            flag_add_label = True
            for anomaly_time_index in anomaly_time_indexes:
                if flag_add_label:
                    axs[0].axvspan(anomaly_time_index - 10, anomaly_time_index + 10,
                                   color='red', alpha=0.25, label=r'True anomalies: $20$-point centered windows')
                    flag_add_label = False
                else:
                    axs[0].axvspan(anomaly_time_index - 10, anomaly_time_index + 10, color='red', alpha=0.25)
            axs[0].plot(time_series, 'k-', markersize=7, label='Time series', zorder=1)
            axs[0].legend(bbox_to_anchor=(0.1, 0.95), loc='upper left', borderaxespad=0., fontsize=16)
            axs[0].tick_params(axis='both', which='major', labelsize=14)

            axs[1].set_frame_on(False)
            axs[1].grid(True)
            axs[1].set_xlim(0, 1000)
            axs[1].set_ylim(0, 1.4)
            axs[1].axvspan(0, self.time_series_length / 2, color='gray', alpha=0.1)
            axs[1].axvspan(self.time_series_length / 2, self.time_series_length, color='green', alpha=0.1)
            axs[1].plot(scores, 'b-', label='Anomaly scores', zorder=1)
            for anomaly_time_index in anomaly_time_indexes:
                axs[1].axvspan(anomaly_time_index - 10, anomaly_time_index + 10, color='red', alpha=0.25)
            anomaly_time_indexes = np.array(self.time_indexes)[anomalies]
            axs[1].scatter(x=anomaly_time_indexes, y=scores[anomaly_time_indexes], s=75, edgecolors='red',
                           facecolor="red", marker='o', label='Detected anomalies', zorder=2)

            axs[1].set_xlabel(r'Time index $n$', fontsize=22)
            axs[0].set_ylabel(r'Observation $y\check_{n}$', color='k', fontsize=22)
            axs[1].set_ylabel(r'Score $\Sigma^{ad}_{n}$', color='b', fontsize=22)
            axs[1].legend(bbox_to_anchor=(0.1, 0.95), loc='upper left', borderaxespad=0., fontsize=16)
            axs[1].tick_params(axis='both', which='major', labelsize=14)

            plt.tight_layout()
            plt.show(block=True)

        if self.plot_statistics:
            plt.subplots(constrained_layout=True)
            plt.axis('off')
            ax = plt.subplot(611)
            plt.plot(time_series, 'bo--', label='Time-series')
            plt.plot(predictions, 'm.--', label='Predictions')
            ax.add_patch(plt.Rectangle((0, 0), self.time_series_length / 2, 100, facecolor="red", alpha=0.1))
            ax.add_patch(
                plt.Rectangle((self.time_series_length / 2, 0), self.time_series_length / 2, 100, facecolor="green",
                              alpha=0.1))
            plt.xticks(np.arange(0, self.time_series_length, step=self.time_series_length / 20))
            plt.grid(axis='x', color='0.95')
            plt.title(time_series_name)
            plt.legend(loc="upper right")
            ax = plt.subplot(612)
            plt.yticks([1.0, 0.0], ["True", "False"])
            cmap = clrs.ListedColormap(['green', 'red'])
            plt.scatter(x=self.time_indexes, y=ground_truth, c=ground_truth.astype(float), marker='d', cmap=cmap)
            ax.add_patch(plt.Rectangle((0, 0), self.time_series_length / 2, 1, facecolor="red", alpha=0.1))
            ax.add_patch(
                plt.Rectangle((self.time_series_length / 2, 0), self.time_series_length / 2, 1, facecolor="green",
                              alpha=0.1))
            plt.xticks(np.arange(0, self.time_series_length, step=self.time_series_length / 20))
            plt.grid(axis='x', color='0.95')
            plt.title('Ground-truth')
            ax = plt.subplot(613)
            plt.yticks([1.0, 0.0], ["True", "False"])
            cmap = clrs.ListedColormap(['green', 'red'])
            plt.scatter(x=self.time_indexes, y=anomalies, c=anomalies.astype(float), marker='d', cmap=cmap)
            ax.add_patch(plt.Rectangle((0, 0), self.time_series_length / 2, 1, facecolor="red", alpha=0.1))
            ax.add_patch(
                plt.Rectangle((self.time_series_length / 2, 0), self.time_series_length / 2, 1, facecolor="green",
                              alpha=0.1))
            plt.xticks(np.arange(0, self.time_series_length, step=self.time_series_length / 20))
            plt.grid(axis='x', color='0.95')
            plt.title('Anomaly indicator')
            ax = plt.subplot(614)
            plt.plot(100 * scores, 'b-')
            ax.add_patch(plt.Rectangle((0, 0), self.time_series_length / 2, 100, facecolor="red", alpha=0.1))
            ax.add_patch(
                plt.Rectangle((self.time_series_length / 2, 0), self.time_series_length / 2, 100, facecolor="green",
                              alpha=0.1))
            plt.xticks(np.arange(0, self.time_series_length, step=self.time_series_length / 20))
            plt.grid(axis='x', color='0.95')
            plt.title('Anomaly score (%)')
            ax = plt.subplot(615)
            plt.plot(layer0_memory_stats, 'r-', label='Layer 0')
            plt.plot(layer1_memory_stats, 'g-', label='Layer 1')
            plt.plot(layer2_memory_stats, 'b-', label='Layer 2')
            ax.add_patch(plt.Rectangle((0, 0), self.time_series_length / 2, 100, facecolor="red", alpha=0.1))
            ax.add_patch(
                plt.Rectangle((self.time_series_length / 2, 0), self.time_series_length / 2, 100, facecolor="green",
                              alpha=0.1))
            plt.xticks(np.arange(0, self.time_series_length, step=self.time_series_length / 20))
            plt.grid(axis='x', color='0.95')
            plt.title('Memory usage (KB)')
            plt.legend(loc="upper left")
            ax = plt.subplot(616)
            # ax = plt.gca()
            ax.set_xlim([0, self.time_series_length])
            ax.set_ylim([0, 1024])
            plt.plot(predicted_observation_symbols, 'ro-', label='Predicted observation symbol')
            plt.plot(observation_symbols, 'g.-', label='Observed symbol')
            plt.plot(predicted_state_symbols, 'b.-', label='Predicted state symbol')
            ax.add_patch(plt.Rectangle((0, 0), self.time_series_length / 2, 2048, facecolor="red", alpha=0.1))
            ax.add_patch(
                plt.Rectangle((self.time_series_length / 2, 0), self.time_series_length / 2, 2048, facecolor="green",
                              alpha=0.1))
            plt.xticks(np.arange(0, self.time_series_length, step=self.time_series_length / 20))
            plt.grid(axis='x', color='0.95')
            plt.title('Predicted symbols')
            plt.legend(loc="upper right")
            plt.show(block=True)

        self.detector = None
        self.test_statistics = None

    def runTest(self, time_series, ground_truth, time_series_name):
        # Initialize the output vectors
        anomalies = np.zeros(self.time_series_length, order='C', dtype=bool)
        scores = np.zeros(self.time_series_length, order='C', dtype=float)
        predictions = np.zeros(self.time_series_length, order='C', dtype=float)
        predicted_observation_symbols = np.zeros(self.time_series_length, order='C', dtype=int)
        observation_symbols = np.zeros(self.time_series_length, order='C', dtype=int)
        predicted_state_symbols = np.zeros(self.time_series_length, order='C', dtype=int)

        # Memory statistics
        layer0_memory_stats = np.zeros(self.time_series_length, order='C', dtype=float)
        layer1_memory_stats = np.zeros(self.time_series_length, order='C', dtype=float)
        layer2_memory_stats = np.zeros(self.time_series_length, order='C', dtype=float)

        # Initialize the detector
        self.detector.initialize(np.min(time_series), np.max(time_series), np.size(time_series))

        # Run the detector
        elapsed_time = 0
        max_detect_time = 0
        for n in self.time_indexes:
            value = time_series[n]
            t = time.time()
            (anomaly,
             score,
             predicted_value,
             predicted_observation_symbol,
             observation_symbol,
             predicted_state_symbol,) = self.detector.handle(value)
            detect_time = time.time() - t
            elapsed_time += detect_time
            if detect_time > max_detect_time:
                max_detect_time = detect_time
            anomalies[n] = anomaly
            scores[n] = score
            predictions[n] = predicted_value
            predicted_observation_symbols[n] = predicted_observation_symbol
            observation_symbols[n] = observation_symbol
            predicted_state_symbols[n] = predicted_state_symbol
            layer0_memory_stats[n], layer1_memory_stats[n], layer2_memory_stats[n] = self.detector.memory_size()
            self.detector.debug()

        # Collect the statistics
        self.test_statistics["elapsed_detect_time"] = elapsed_time
        self.test_statistics["mean_detect_time"] = elapsed_time / self.time_series_length
        self.test_statistics["max_detect_time"] = max_detect_time
        self.test_statistics["time_series_name"] = time_series_name
        self.test_statistics["time_series"] = time_series
        self.test_statistics["ground_truth"] = ground_truth
        self.test_statistics["anomalies"] = anomalies
        self.test_statistics["scores"] = scores
        self.test_statistics["predictions"] = predictions
        self.test_statistics["predicted_observation_symbols"] = predicted_observation_symbols
        self.test_statistics["observation_symbols"] = observation_symbols
        self.test_statistics["predicted_state_symbols"] = predicted_state_symbols
        self.test_statistics["layer0_memory_stats"] = layer0_memory_stats
        self.test_statistics["layer1_memory_stats"] = layer1_memory_stats
        self.test_statistics["layer2_memory_stats"] = layer2_memory_stats
        self.test_statistics["total_memory_stats"] = layer0_memory_stats[-1] + \
                                                     layer1_memory_stats[-1] + \
                                                     layer2_memory_stats[-1]

    def test_0_cte_time_series_with_spike_anomalies(self):
        """
        Test case 0: Constant valued time-series with abnormal spikes.
        """

        # Time-series parameters
        time_series_name = 'Constant-valued time-series with random spikes'
        time_series_cte_value = 15
        number_of_spikes = 3
        spike_value = 100
        seed = 0

        np.random.seed(seed)

        # Build the time-series
        spike_locations = np.unique(np.random.choice(self.anomaly_indexes, number_of_spikes, replace=True))
        time_series = time_series_cte_value * np.ones(self.time_series_length, order='C', dtype=float)
        time_series[spike_locations] = spike_value

        # Build the ground_truth
        ground_truth = np.zeros(self.time_series_length, order='C', dtype=bool)
        ground_truth[spike_locations] = True

        # Run the test
        self.runTest(time_series, ground_truth, time_series_name)

    def test_1_sin_time_series_with_spike_anomalies(self):
        """
        Test case 1: Sinusoidal time-series with abnormal spikes.
        """

        # Time-series parameters
        time_series_name = 'Sinusoidal time-series with random spikes'
        time_series_amplitude = 15
        time_series_offset = 15
        time_series_frequency = 0.5  # [Hz]
        time_series_phase = 0  # [rad]
        number_of_spikes = 3
        spike_value = 100
        seed = 1

        np.random.seed(seed)

        # Build the time-series
        spike_locations = np.unique(np.random.choice(self.anomaly_indexes, number_of_spikes, replace=True))
        time_series = time_series_offset + time_series_amplitude * \
                      np.sin(2 * np.pi * time_series_frequency * self.time_instants + time_series_phase)
        time_series[spike_locations] = spike_value

        # Build the ground_truth
        ground_truth = np.zeros(self.time_series_length, order='C', dtype=bool)
        ground_truth[spike_locations] = True

        # Run the test
        self.runTest(time_series, ground_truth, time_series_name)

    def test_2_squared_time_series_with_spike_anomalies(self):
        """
        Test case 2: Squared time-series with abnormal spikes.
        """

        # Time-series parameters
        time_series_name = 'Squared time-series with random spikes'
        time_series_amplitude = 15
        time_series_offset = 15
        time_series_frequency = 0.5  # [Hz]
        time_series_phase = 0  # [rad]
        number_of_spikes = 2
        spike_value = 100
        seed = 0

        np.random.seed(seed)

        # Build the time-series
        spike_locations = np.unique(np.random.choice(self.anomaly_indexes, number_of_spikes, replace=True))
        time_series = time_series_offset + time_series_amplitude * \
                      np.sign(np.sin(2 * np.pi * time_series_frequency * self.time_instants + time_series_phase))
        time_series[spike_locations] = spike_value

        # Build the ground_truth
        ground_truth = np.zeros(self.time_series_length, order='C', dtype=bool)
        ground_truth[spike_locations] = True

        # Run the test
        self.runTest(time_series, ground_truth, time_series_name)

    def test_3_sin_time_series_without_anomalies(self):
        """
        Test case 3: Sinusoidal time-series without anomalies.
        """

        # Time-series parameters
        time_series_name = 'Sinusoidal time-series without anomalies'
        time_series_amplitude = 15
        time_series_offset = 15
        time_series_frequency = 0.1  # [Hz]
        time_series_phase = 0  # [rad]
        seed = 1

        np.random.seed(seed)

        # Build the time-series
        time_series = time_series_offset + time_series_amplitude * \
                      np.sin(2 * np.pi * time_series_frequency * self.time_instants + time_series_phase)

        # Build the ground_truth
        ground_truth = np.zeros(self.time_series_length, order='C', dtype=bool)

        # Run the test
        self.runTest(time_series, ground_truth, time_series_name)

    def test_4_squared_time_series_without_anomalies(self):
        """
        Test case 4: Squared time-series without anomalies.
        """

        # Time-series parameters
        time_series_name = 'Squared time-series without anomalies'
        time_series_amplitude = 40
        time_series_offset = 40
        time_series_frequency = 0.5  # [Hz]
        time_series_phase = 0  # [rad]
        seed = 0

        np.random.seed(seed)

        # Build the time-series
        time_series = time_series_offset + time_series_amplitude * \
                      np.sign(np.sin(2 * np.pi * time_series_frequency * self.time_instants + time_series_phase))

        # Build the ground_truth
        ground_truth = np.zeros(self.time_series_length, order='C', dtype=bool)

        # Run the test
        self.runTest(time_series, ground_truth, time_series_name)

    def test_5_saw_time_series_with_spike_anomalies(self):
        """
        Test case 5: Sawtooth time-series with abnormal spikes.
        """

        # Time-series parameters
        time_series_name = 'Sinusoidal time-series with random spikes'
        time_series_amplitude = 15
        time_series_offset = 15
        time_series_frequency = 0.5  # [Hz]
        time_series_phase = 0  # [rad]
        time_series_noise_std_dev = 1
        number_of_spikes = 3
        spike_value = 100
        seed = 3

        np.random.seed(seed)

        # Build the time-series
        spike_locations = np.unique(np.random.choice(self.anomaly_indexes, number_of_spikes, replace=True))
        time_series = time_series_offset + time_series_amplitude * \
                      signal.sawtooth(2 * np.pi * time_series_frequency * self.time_instants + time_series_phase,
                                      width=0.5)
        time_series[spike_locations] = spike_value

        # Build the ground_truth
        ground_truth = np.zeros(self.time_series_length, order='C', dtype=bool)
        ground_truth[spike_locations] = True

        # Run the test
        self.runTest(time_series, ground_truth, time_series_name)


if __name__ == '__main__':
    unittest.main()

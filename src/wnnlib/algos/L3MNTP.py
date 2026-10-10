import numpy as np
from wnnlib.vgram import VGRAMArray, VGRAMNode
from wnnlib.codecs import FixedScalarCodec, AdaptiveScalarCodec
from wnnlib.utils.BitUtils import BitUtils
import gc


class L3MNTP:
    """
    This class implements a continuous learning (CL), next token prediction (NTP) for streamed sequences of tokens
    using a non-parametric Bayesian procedure to build a suitable model of the underlying stochastic process emitting
    the tokens.

    Inefficient, slightly pretty, hopefully correct Pythonic (yuck) implementation of the LLM-NTP predictor.
    """

    def __init__(self, cs, ps, alphas,
                 az, bz, cz, dz, pz, alphaz,
                 test=1, delta=0, tau=0.75,
                 learning_rate=1, sub_seq_len=4):
        """
        Initialize the LLM-NTP predictor using the nine hyperparameters.

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


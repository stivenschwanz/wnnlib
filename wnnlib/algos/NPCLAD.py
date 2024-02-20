from numpy import random
import numpy as np
from wnnlib.vgram_array import VGRAMArray, VGRAMNode
from nupic.encoders import AdaptiveScalarEncoder


class NPCLAD:
    """
    This class implements a continuous learning (CL), anomaly detector (AD) for streamed sequences of observations
    using a non-parametric Bayesian procedure to build a suitable model of the underlying stochastic process emitting
    the observations.
    """
    def __init__(self, cs=2^11, ps=2^15, alphas=2^-10, az=2^6, bz=2^4, cz=2^11, dz=2^11, pz=2^5, alphaz=2^-10, test=1):
        """
        Initialize the nP-CLAD detector using the nine hyper-parameters.

        Parameters:
            cs (int): Maximum number of distinct hidden vectors (symbols) in $ \boldsymbol{\Omega}_{s}} $.
            ps (int): Maximum number of equiprobable hypothesis in the belief $ {\bf b}_{n|n-1} $, $ \forall n > 0 $.
            alphas (float): Pseudo-count parameter for the prior Dirichlet distribution $ Dir(c_{s}; \boldsymbol{\alpha}_{0} $.
            az (int): Number of activated bits of the sparsely encoded observations.
            bz (int): Minimum distance between sparsely encoded observations.
            cz (int): Maximum number of distinct encoded observations (symbols) in $ \boldsymbol{\Omega}_{z}} $.
            dz (int): Total number of bits of the sparsely encoded observations.
            pz (int): Maximum number of equiprobable hypothesis in the belief $ \tilde{\bf b}_{n|n-1} $, $ \forall n > 0 $.
            alphaz (float): Pseudo-count parameter for the prior Dirichlet distribution $ Dir(c_{z}; \tilde\boldsymbol{\alpha}_{0} $.
            test (int): AD test type
        """

        # Save the parameters
        self.cs = cs
        self.ps = ps
        self.alphas = alphas * np.ones(cs, order='C', dtype=float)
        self.az = az
        self.bz = bz
        self.cz = cz
        self.dz = dz
        self.pz = pz
        self.alphaz = alphaz * np.ones(cz, order='C', dtype=float)
        self.test = test

        # Initialize members
        self.bs_n_n_1 = None
        self.bs_n_1_n = None
        self.bz_n_1_n = None
        self.wnn_layer0 = None
        self.wnn_layer1 = None
        self.wnn_layer2 = None
        self.zn = None
        self.encoder = None

    def __del__(self):
        """
        Delete method. The garbage collector will hopefully work here.
        """

        # Clean up parameters
        self.cs = None
        self.ps = None
        self.alphas = None
        self.az = None
        self.bz = None
        self.cz = None
        self.dz = None
        self.pz = None
        self.alphaz = None
        self.test = None

        # Clean up members
        self.bs_n_n_1 = None
        self.bs_n_1_n = None
        self.bz_n_1_n = None
        self.wnn_layer0 = None
        self.wnn_layer1 = None
        self.wnn_layer2 = None
        self.zn = None
        self.encoder = None

    def handle(self, y_n_1):
        """
        Handle the current encoded observation.

        Parameters:
            y_n_1 (float[]): Observation vector at instant $ n + 1 $.

        Returns:
            (bool): True if the current observation is an anomaly, false otherwise.
            (float): Anomaly score indicating how likely the current observation is an anomaly.
        """
        # Encode observation
        z_n_1, k = self.encode(y_n_1)

        # Predict next observation
        z_n_1_n, probz_n_1_n = self.predict()

        detection = (False, 0.0)
        # Detect anomaly
        if self.test == 1:
            detection = (np.count_nonzero(z_n_1_n != z_n_1) == 0, probz_n_1_n)
        elif self.test == 2:
            detection = (self.bz_n_1_n[k] == 1, 1/np.count_nonzero(self.bz_n_1_n))
        else:
            raise Exception("Invalid AD test type.")

        # Update model
        self.learn(z_n_1, k)

        return detection

    def initialize(self):
        """
        Initialize the filter.
        """

        # ----------------------------------------------------------------------
        # Draw the prior posterior $ \\boldsymbol{\pi}_{n|n-1} $ from the Dirichlet
        # distribution $ Dir \left( c_{s}; \\boldsymbol{\\alpha}_{0} \right) $
        # ----------------------------------------------------------------------
        pis_0 = np.random.dirichlet(self.alphas)

        # ----------------------------------------------------------------------
        # Draw up to $ p_{s} $ unique samples from the prior posterior $ \\boldsymbol{\pi}_{0|-1} $
        # ----------------------------------------------------------------------
        ells = np.unique(np.random.choice(self.cs, size=self.ps, replace=True, p=pis_0))

        # ----------------------------------------------------------------------
        # Build the prior belief $ {\\bf b}_{0|-1} $
        # ----------------------------------------------------------------------
        self.bs_n_n_1 = np.zeros(self.cs, order='C', dtype=bool)
        self.bs_n_n_1[ells] = True

        # ----------------------------------------------------------------------
        # Initialize the other beliefs
        # ----------------------------------------------------------------------
        self.bs_n_1_n = np.zeros(self.cs, order='C', dtype=bool)
        self.bz_n_1_n = np.zeros(self.cz, order='C', dtype=bool)

        # ----------------------------------------------------------------------
        # Initialize layers
        # ----------------------------------------------------------------------
        self.wnn_layer0 = VGRAMNode.VGRAMNode(pattern_length=self.dz,
                                              min_mem_size=1, max_mem_size=2 ^ 14,
                                              min_dist=2, max_dist=8)
        self.wnn_layer1 = VGRAMArray.VGRAMArray(output_dims=(self.cs, 1), pattern_length=self.dz + self.cs,
                                                min_mem_size=1, max_mem_size=2 ^ 14,
                                                min_dist=2, max_dist=8)
        self.wnn_layer2 = VGRAMArray.VGRAMArray(output_dims=(self.cz, 1), pattern_length=self.cs,
                                                min_mem_size=1, max_mem_size=2 ^ 14,
                                                min_dist=0, max_dist=0)

        # ----------------------------------------------------------------------
        # Initialize the previous encoded observation
        # ----------------------------------------------------------------------
        self.zn = np.zeros(self.dz, order='C', dtype=bool)
        jz = np.unique(np.random.choice(self.dz, size=self.az, replace=True))
        self.zn[jz] = True
        self.wnn_layer2.learn(self.zn)

        # ----------------------------------------------------------------------
        # Initialize the adaptive scalar encoder
        # ----------------------------------------------------------------------
        self.encoder = AdaptiveScalarEncoder(w=130, n=self.dz)

    def predict(self):
        """
        Predict the next encoded observation.

        Returns:
            (bool[]): Predicted encoded observation vector $ \\hat{\bf z}_{n+1|n} = \\boldsymbol{\\mu}^{(\\hat{k})} $
            (float): Probability of the predicted symbol $ {\\tilde{\\mu}}^{(\\hat{k})}_{n+1|n} $.
        """

        # ----------------------------------------------------------------------
        # Step 1: Retrieve hyper-parameters $ \\boldsymbol{\\alpha}_{n|n-1} $
        # indexed by $ \\check{\bf z}_{n} $ and $ {\\bf b}_{n|n-1} $
        # ----------------------------------------------------------------------
        alphas_n_n_1 = self.wnn_layer1.recall(np.concatenate((self.z_n, self.bs_n_n_1)))

        # ----------------------------------------------------------------------
        # Step 2: Draw the predicted posterior $ \\boldsymbol{\pi}_{n+1|n} $ from
        # the Dirichlet distribution $ Dir \left( c_{s}; \\boldsymbol{\\alpha}_{n|n-1} \right) $
        # ----------------------------------------------------------------------
        pis_n_1_n = np.random.dirichlet(alphas_n_n_1)

        # ----------------------------------------------------------------------
        # Step 3: Draw up to $ p_{s} $ unique samples from the predicted posterior $ \\boldsymbol{\pi}_{n+1|n} $
        # ----------------------------------------------------------------------
        ells = np.unique(np.random.choice(self.cs, size=self.ps, replace=True, p=pis_n_1_n))

        # ----------------------------------------------------------------------
        # Step 4: Build the predicted belief $ {\\bf b}_{n+1|n} $
        # ----------------------------------------------------------------------
        self.bs_n_1_n = np.zeros(self.cs, order='C', dtype=bool)
        self.bs_n_1_n[ells] = True

        # ----------------------------------------------------------------------
        # Step 5: Retrieve the hyper-parameters $ \\tilde{\\boldsymbol{\\alpha}}_{n|n-1} $
        # indexed by $ {\\bf b}_{n+1|n} $
        # ----------------------------------------------------------------------
        alphaz_n_n_1 = self.wnn_layer2.recall(self.bs_n_1_n)

        # ----------------------------------------------------------------------
        # Step 6: Draw the predicted posterior $ \tilde{\\boldsymbol{\pi}}_{n+1|n} $
        # from the Dirichlet distribution $ Dir \left( c_{z}; \tilde{\\boldsymbol{\\alpha}}_{n|n-1} \right) $
        # ----------------------------------------------------------------------
        piz_n_1_n = np.random.dirichlet(alphaz_n_n_1)

        # ----------------------------------------------------------------------
        # Step 7: Draw up to $ p_{z} $ unique samples from the predicted posterior $ \tilde{\\boldsymbol{\pi}}_{n+1|n} $
        # ----------------------------------------------------------------------
        ellz = np.unique(np.random.choice(self.cz, size=self.pz, replace=True, p=piz_n_1_n))

        # ----------------------------------------------------------------------
        # Step 8: Build the predicted belief $ \\tilde{\\bf b}_{n+1|n} $
        # ----------------------------------------------------------------------
        self.bz_n_1_n = np.zeros(self.cz, order='C', dtype=bool)
        self.bz_n_1_n[ellz] = True

        # ----------------------------------------------------------------------
        # Step 9: Predict the next observation $ \hat{\bf z}_{n+1|n} $
        # ----------------------------------------------------------------------
        k = np.argmax(piz_n_1_n)
        z_n_1_n = self.wnn_layer0.get_pattern_by_index(k)
        probz_n_1_n = piz_n_1_n[k]

        return z_n_1_n, probz_n_1_n

    def learn(self, z_n_1, k):
        """
        Predict the next encoded observation.

        Parameters:
            z_n_1 (bool[]): Sparsely encoded observation vector $ \\check{\bf z}_{n+1} $ at instant $ n+1 $.
            k (int): Index of the sparse encoded observation.
        """

        # ----------------------------------------------------------------------
        # Step 1: Check if there is a match
        # ----------------------------------------------------------------------
        if self.bz_n_1_n[k] == 1:
            # ----------------------------------------------------------------------
            # Step 2: Retrieve the hyper-parameters $ \\boldsymbol{\\alpha}_{n|n-1} $
            # indexed by $ \\check{\bf z}_{n} $ and $ {\\bf b}_{n|n-1} $
            # ----------------------------------------------------------------------
            alphas_n_n_1 = self.wnn_layer1.recall(np.concatenate((self.z_n, self.bs_n_n_1)))

            # ----------------------------------------------------------------------
            # Step 3: Update the hyper-parameters $ \\boldsymbol{\\alpha}_{n|n-1} $
            # ----------------------------------------------------------------------
            alphas_n_1_n = alphas_n_n_1 + self.bs_n_1_n

            # ----------------------------------------------------------------------
            # Step 4: Retrieve the hyper-parameters $ \\tilde{\\boldsymbol{\\alpha}}_{n|n-1} $
            # indexed by $ {\\bf b}_{n+1|n} $
            # ----------------------------------------------------------------------
            alphaz_n_n_1 = self.wnn_layer2.recall(self.bs_n_1_n)

            # ----------------------------------------------------------------------
            # Step 5: Update the hyper-parameters $ \\tilde{\\boldsymbol{\\alpha}}_{n|n-1} $
            # ----------------------------------------------------------------------
            alphaz_n_1_n = alphaz_n_n_1 + self.bz_n_1_n
            alphaz_n_1_n[k] += 1
        else:  # Step 6...
            # ----------------------------------------------------------------------
            # Steps 7,8: Initialize the hyper-parameters $ \\boldsymbol{\\alpha}_{n+1|n} $
            # ----------------------------------------------------------------------
            alphas_n_1_n = self.alphas

            # ----------------------------------------------------------------------
            # Steps 9,10: Initialize the hyper-parameters $ \\tilde{\\boldsymbol{\\alpha}}_{n+1|n} $
            # ----------------------------------------------------------------------
            alphaz_n_1_n = self.alphaz
        # Step 11...

        # ----------------------------------------------------------------------
        # Steps 12,13: Store new hyper-parameters
        # ----------------------------------------------------------------------
        self.wnn_layer1.learn(np.concatenate((self.z_n, self.bs_n_n_1)), alphas_n_1_n)
        self.wnn_layer2.learn(self.bs_n_1_n, alphaz_n_1_n)

        # Update the previous observation
        self.zn = z_n_1

    def encode(self, y_n_1):
        """
        Encode the incoming observation at instant $ n + 1 $.

        Parameters:
             y_n_1 (float[]): Observation vector at instant $ n + 1 $.
        Return:
            (bool[]): Sparsely encoded observation vector $ \\check{\bf z}_{n+1} $ at instant $ n+1 $.
            (int): Index of the observation symbol in $ \\boldsymbol{\\Omega}_{z} $
        """

        z_n_1 = self.encoder.encode(y_n_1)

        [d, k] = self.wnn_layer0.find_closest_pattern(z_n_1)

        if d != 0:
            k = self.wnn_layer0.learn(z_n_1)

        return z_n_1, k

    def debug(self, observation=None, output_value=None):
        """
        Debug node memory.
        """




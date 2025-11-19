import numpy as np
from matplotlib import pyplot as plt


class TimeCausGaborCodec:
    omega: np.ndarray
    freqs: np.ndarray
    mu: np.ndarray
    delta_t: float
    c: float
    N: float
    J: int
    K: int
    level: np.ndarray
    level_prev: np.ndarray
    n: int
    L: int
    k: int

    def __init__(self, freqs, delta_t=1.0, c=2.0, numlevels=16, N=8.0, L=100, number_of_active_bits=8):

        # Distribution parameter
        self.c = c

        assert self.c > 1.0

        # Number of temporal scale levels
        self.K = numlevels

        assert self.K > 1

        # Proportionality factor
        self.N = N

        assert self.N >= 1

        # Angular frequencies [Hz]
        self.freqs = np.array(freqs)

        self.J = np.size(self.freqs)

        assert self.J > 1

        # Compute angular frequencies [rad/s]
        self.omega = 2.0 * np.pi * self.freqs

        # Initialize the output buffers
        self.level = np.zeros((self.J, self.K, 2), order='C', dtype=float)
        self.level_prev = np.zeros((self.J, self.K, 2), order='C', dtype=float)

        # Sampling time [s]
        self.delta_t = delta_t

        assert self.delta_t > 0.0

        # Sampling rate [Hz]
        r = 1.0 / self.delta_t

        # Initialize the time constants
        self.mu = np.zeros((self.J, self.K))
        self.gain = np.zeros((self.J, self.K))

        for j in range(self.J):
            sigma_j0 = 2.0 * np.pi * self.N / self.omega[j]
            tau_jref = (r * sigma_j0)**2.0

            tau_jk_1 = 0
            for k in range(self.K):
                # Compute the temporal scale levels
                tau_jk = self.c**(2.0 * (k + 1 - self.K)) * tau_jref

                # Compute the temporal scale increments
                delta_tau_jk = tau_jk - tau_jk_1
                tau_jk_1 = tau_jk

                # Compute the time constant
                self.mu[j, k] = 0.5 * (np.sqrt(1.0 + 4.0 * delta_tau_jk) - 1.0)

                # Set the gain
                self.gain[j, k] = 1.0 / (1.0 + self.mu[j, k])

        self.n = 0

        # Signal length
        self.L = L

        assert self.L > 1

        self.duration = self.L * self.delta_t

        # Number of active bits
        self.k = number_of_active_bits

        assert 0 < self.k < self.J * self.K

        self.spectrogramchart = np.zeros((self.J * self.K, self.L+1))

        self.encodegramchart = np.zeros((2*self.J * self.K, self.L+1))

    def update(self, signal):
        for j in range(self.J):
            omega_jt = self.omega[j] * self.n * self.delta_t
            level_j_k_1 = np.array([signal * np.cos(omega_jt), -signal * np.sin(omega_jt)])
            for k in range(self.K):
                level_prev_j_k = self.level_prev[j, k, :]
                self.level[j, k, :] = level_prev_j_k + self.gain[j, k] * (level_j_k_1 - level_prev_j_k)
                level_j_k_1 = self.level[j, k, :]
        self.level_prev = self.level.copy()
        self.n += 1
        # return abs_level
        return self.level

    def encode(self, signal):
        level_flattened = self.update(signal).flatten(order='C')
        ind = np.argpartition(level_flattened, -self.k)[-self.k:]  # index of the k highest elements
        z = np.zeros(level_flattened.shape, order='C', dtype=bool)
        z[ind] = True

        self.encodegramchart[:, self.n - 1] = z

        plt.clf()
        plt.imshow(self.encodegramchart, extent=[0, self.delta_t*self.L, 0, self.J*self.K])
        plt.xlabel("Time (seconds)")
        plt.ylabel("Log Frequency (Hz)")
        plt.show(block=False)
        plt.pause(0.000000001)

        return z, ind

    def spectrogram(self, signal, lowsoftthresh: float = 0.000001, maxrange: float = 60):
        level = self.update(signal)
        abs_level = np.sqrt(level[:, :, 0] ** 2 + level[:, :, 1] ** 2).flatten(order='C')

        # Compute logarithmic magnitudes in dB, with additional lower bound
        maxval = np.max(abs_level)
        logspectrogram = 20 * np.log10(abs_level / maxval + lowsoftthresh)
        logspectrogram[logspectrogram < -maxrange] = -maxrange

        self.spectrogramchart[:, self.n - 1] = logspectrogram
        plt.clf()

        im = \
            plt.imshow(self.spectrogramchart, \
                       cmap='jet', interpolation='nearest', aspect='auto', \
                       #origin='lower', \
                       extent=[0, self.duration, min(self.freqs), max(self.freqs)])
        plt.colorbar(im)

        plt.xlabel("Time (seconds)")
        plt.ylabel("Log Frequency (Hz)")
        plt.show(block=False)
        plt.pause(0.000000001)

        return logspectrogram

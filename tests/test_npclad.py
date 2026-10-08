import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import unittest
import time
import sys
from wnnlib.algos.NPCLAD import NPCLAD
from scipy import signal


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
    test_idx = None

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

            if "pytest" in sys.modules:
                plt.savefig(f"./figs/test_npclad_pretty_{self.test_idx}.png", bbox_inches='tight')
            else:
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
            ax.set_xlim((0, self.time_series_length))
            ax.set_ylim((0, 1024))
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

            if "pytest" in sys.modules:
                plt.savefig(f"./figs/test_npclad_statistics_{self.test_idx}.png", bbox_inches='tight')
            else:
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
        self.test_idx = 0

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
        self.test_idx = 1

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
        self.test_idx = 2

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
        self.test_idx = 3

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
        self.test_idx = 4

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
        self.test_idx = 5

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

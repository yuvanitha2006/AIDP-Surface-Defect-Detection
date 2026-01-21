import numpy as np


class DriftDetector:
    """
    Statistical drift detector using Population Stability Index (PSI)
    """

    def __init__(self, baseline_window=200, test_window=200, psi_threshold=0.2):
        """
        Args:
            baseline_window: size of reference window
            test_window: size of current window
            psi_threshold: drift sensitivity
        """
        self.baseline_window = baseline_window
        self.test_window = test_window
        self.psi_threshold = psi_threshold

    def _calculate_psi(self, expected, actual, bins=10):
        """
        Compute Population Stability Index
        """
        breakpoints = np.linspace(0, 100, bins + 1)
        expected_perc = np.percentile(expected, breakpoints)
        actual_perc = np.percentile(actual, breakpoints)

        psi = 0.0
        for i in range(len(expected_perc) - 1):
            exp_count = np.mean(
                (expected >= expected_perc[i]) & (expected < expected_perc[i + 1])
            )
            act_count = np.mean(
                (actual >= actual_perc[i]) & (actual < actual_perc[i + 1])
            )

            exp_count = max(exp_count, 1e-6)
            act_count = max(act_count, 1e-6)

            psi += (act_count - exp_count) * np.log(act_count / exp_count)

        return psi

    def detect(self, scores):
        """
        Detect drift based on anomaly scores
        """
        drift_flags = []
        psi_values = []

        for i in range(self.baseline_window + self.test_window, len(scores)):
            baseline = scores[i - self.baseline_window - self.test_window : i - self.test_window]
            current = scores[i - self.test_window : i]

            psi = self._calculate_psi(baseline, current)
            psi_values.append(psi)

            drift_flags.append(psi > self.psi_threshold)

        return np.array(drift_flags), np.array(psi_values)

import numpy as np
from enum import Enum


class Decision(Enum):
    NORMAL = "normal"
    MONITOR = "monitor"
    HUMAN_REVIEW = "human_review"
    CRITICAL_ALERT = "critical_alert"


class DecisionEngine:
    """
    Risk-aware decision engine for anomaly detection
    """

    def __init__(
        self,
        anomaly_threshold,
        uncertainty_threshold,
        critical_threshold,
    ):
        """
        Args:
            anomaly_threshold: base anomaly threshold (e.g., 95th percentile)
            uncertainty_threshold: uncertainty above which human review is required
            critical_threshold: very high anomaly score threshold
        """
        self.anomaly_threshold = anomaly_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.critical_threshold = critical_threshold

    def decide(self, anomaly_score, uncertainty):
        """
        Make a decision based on anomaly score and uncertainty
        """

        if anomaly_score > self.critical_threshold and uncertainty < self.uncertainty_threshold:
            return Decision.CRITICAL_ALERT

        if anomaly_score > self.anomaly_threshold and uncertainty >= self.uncertainty_threshold:
            return Decision.HUMAN_REVIEW

        if anomaly_score > self.anomaly_threshold:
            return Decision.MONITOR

        return Decision.NORMAL

    def batch_decide(self, scores, uncertainties):
        """
        Vectorized decisions for all windows
        """
        decisions = []

        for s, u in zip(scores, uncertainties):
            decisions.append(self.decide(s, u))

        return decisions

import numpy as np


class FeedbackLearner:
    """
    Learns from human feedback to adapt system behavior
    """

    def __init__(self, learning_rate=0.05):
        self.learning_rate = learning_rate

    def adjust_thresholds(
        self,
        engine,
        feedback_records,
    ):
        """
        Adapt decision thresholds using feedback
        """

        false_alerts = [
            r for r in feedback_records
            if r["system_decision"] != r["human_decision"]
            and r["system_decision"] == "critical_alert"
        ]

        missed_anomalies = [
            r for r in feedback_records
            if r["system_decision"] == "normal"
            and r["human_decision"] != "normal"
        ]

        if false_alerts:
            engine.critical_threshold *= (1 + self.learning_rate)

        if missed_anomalies:
            engine.anomaly_threshold *= (1 - self.learning_rate)

        return engine

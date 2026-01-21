import json
from pathlib import Path
from datetime import datetime, timezone
from enum import Enum

class FeedbackStore:
    def __init__(self, storage_path="data/feedback/feedback_log.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.storage_path.exists():
            self._safe_write([])

    def _safe_read(self):
        try:
            with open(self.storage_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _safe_write(self, data):
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_feedback(
        self,
        timestamp,
        anomaly_score,
        uncertainty,
        system_decision,
        human_decision,
        comment=""
    ):
        if isinstance(system_decision, Enum):
            system_decision = system_decision.value

        record = {
            "timestamp": timestamp,
            "anomaly_score": float(anomaly_score),
            "uncertainty": float(uncertainty),
            "system_decision": system_decision,
            "human_decision": human_decision,
            "comment": comment,
            "logged_at": datetime.now(timezone.utc).isoformat()
        }

        data = self._safe_read()
        data.append(record)
        self._safe_write(data)

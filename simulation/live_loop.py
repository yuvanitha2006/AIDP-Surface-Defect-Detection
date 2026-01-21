import sys
import time
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from inference.predict import predict_anomaly

print("🚀 Live Anomaly Detection Started...\n")

while True:
    sample = {
        "anomaly_score": round(random.uniform(0.0, 1.0), 3),
        "uncertainty": round(random.uniform(0.0, 0.1), 3)
    }

    result = predict_anomaly(sample)

    print(f"Input: {sample}")
    print(f"Prediction: {result}\n")

    time.sleep(2)

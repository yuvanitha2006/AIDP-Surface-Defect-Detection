import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

# Dummy baseline training data
X = pd.DataFrame({
    "anomaly_score": [0.1, 0.2, 0.15, 0.3, 0.12],
    "uncertainty": [0.02, 0.03, 0.01, 0.04, 0.02]
})

model = IsolationForest(contamination=0.2, random_state=42)
model.fit(X)

joblib.dump(model, "models/model_v1.pkl")
print("✅ Baseline model saved")

from pathlib import Path
import joblib
import numpy as np

# -----------------------------
# LOAD LATEST TRAINED MODEL
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"

model_files = sorted(MODEL_DIR.glob("model_*.pkl"))

if not model_files:
    raise FileNotFoundError("❌ No trained model found in models/ directory")

model_path = model_files[-1]  # latest model
print(f"✅ Loaded model: {model_path.name}")

model = joblib.load(model_path)


# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_anomaly(features: dict):
    """
    features example:
    {
        "anomaly_score": 0.8,
        "uncertainty": 0.02
    }
    """

    X = np.array([[features["anomaly_score"], features["uncertainty"]]])
    prediction = model.predict(X)[0]

    decision = "anomaly" if prediction == 1 else "normal"

    return {
        "prediction": int(prediction),
        "decision": decision
    }

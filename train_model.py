import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from pathlib import Path
import joblib
from datetime import datetime

# -----------------------------
# PATHS
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "processed" / "processed_data.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

FEATURE_COLUMNS = ["anomaly_score", "uncertainty"]

X = df[FEATURE_COLUMNS].values

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = IsolationForest(
    n_estimators=200,
    contamination=0.1,
    random_state=42
)

model.fit(X)

# -----------------------------
# SAVE MODEL
# -----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = MODEL_DIR / f"model_{timestamp}.pkl"

joblib.dump(model, model_path)

print(f"✅ Model saved at: {model_path}")

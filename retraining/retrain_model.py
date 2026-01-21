import joblib
import json
from pathlib import Path
from sklearn.ensemble import IsolationForest
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models"
REGISTRY_PATH = ROOT / "data" / "retraining" / "model_registry.json"

def retrain_model(df):
    X = df[["anomaly_score", "uncertainty"]]

    model = IsolationForest(contamination=0.25, random_state=42)
    model.fit(X)

    version = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    model_path = MODEL_DIR / version

    joblib.dump(model, model_path)

    registry = []
    if REGISTRY_PATH.exists():
        registry = json.load(open(REGISTRY_PATH))

    registry.append({
        "model": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "records": len(df)
    })

    json.dump(registry, open(REGISTRY_PATH, "w"), indent=2)

    return version

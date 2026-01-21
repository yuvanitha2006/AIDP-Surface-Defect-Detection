import json
import pandas as pd
from pathlib import Path

# -----------------------------
# PATHS
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parent
FEEDBACK_PATH = ROOT_DIR / "data" / "feedback" / "feedback_log.json"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "processed_data.csv"

# -----------------------------
# LOAD FEEDBACK
# -----------------------------
if not FEEDBACK_PATH.exists():
    raise FileNotFoundError(f"Feedback file not found at {FEEDBACK_PATH}")

with open(FEEDBACK_PATH, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# -----------------------------
# BASIC VALIDATION
# -----------------------------
required_cols = {"anomaly_score", "uncertainty"}
if not required_cols.issubset(df.columns):
    raise ValueError("Required columns missing in feedback data")

# -----------------------------
# SELECT FEATURES
# -----------------------------
processed_df = df[["anomaly_score", "uncertainty"]]

processed_df.to_csv(OUTPUT_PATH, index=False)

print("✅ processed_data.csv created successfully")
print(OUTPUT_PATH)

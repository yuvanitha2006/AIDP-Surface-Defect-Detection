import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import json
from datetime import datetime, timezone
from scipy.stats import ks_2samp
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


from retraining.retrain_logger import log_retraining_event

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="ADIP – Anomaly Monitoring Dashboard",
    layout="wide"
)

st.title("📊 Autonomous Decision Intelligence Platform")
st.subheader("Anomaly Detection Monitoring Dashboard")

# -----------------------------
# PATH RESOLUTION (CRITICAL)
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
FEEDBACK_PATH = ROOT_DIR / "data" / "feedback" / "feedback_log.json"

RETRAIN_LOG_PATH = ROOT_DIR / "data" / "retraining" / "retrain_log.json"
RETRAIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# LOAD FEEDBACK DATA SAFELY
# -----------------------------
def load_feedback_data(path: Path):
    if not path.exists():
        return pd.DataFrame()

    try:
        with open(path, "r") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except json.JSONDecodeError:
        st.error("⚠️ Feedback file is corrupted.")
        return pd.DataFrame()

df = load_feedback_data(FEEDBACK_PATH)

# -----------------------------
# LOAD RETRAINING HISTORY SAFELY
# -----------------------------
def load_retraining_history(path: Path):
    if not path.exists():
        return []

    try:
        with open(path, "r") as f:
            content = f.read().strip()
            if content == "":
                return []
            return json.loads(content)
    except json.JSONDecodeError:
        return []


# -----------------------------
# EMPTY STATE
# -----------------------------
if df.empty:
    st.warning("No feedback data available yet.")
    st.info("Run the feedback logging code to populate data.")
    st.stop()

# -----------------------------
# DATA PREPROCESSING
# -----------------------------
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

def ks_drift_test(series, ref_size=100, cur_size=50, alpha=0.05):
    """
    Perform KS-test between reference and current windows
    """
    if len(series) < ref_size + cur_size:
        return None, None, None

    reference = series.iloc[-(ref_size + cur_size):-cur_size]
    current = series.iloc[-cur_size:]

    ks_stat, p_value = ks_2samp(reference, current)

    drift = p_value < alpha
    return ks_stat, p_value, drift
# -----------------------------
# ONLINE DRIFT SCORE
# -----------------------------
window = 30  # rolling baseline window

df["rolling_mean"] = df["anomaly_score"].rolling(window).mean()
df["drift_score"] = (df["anomaly_score"] - df["rolling_mean"]).abs()
# Adaptive threshold
drift_mean = df["drift_score"].mean()
drift_std = df["drift_score"].std()

ALERT_THRESHOLD = drift_mean + 2 * drift_std
latest_drift = df["drift_score"].iloc[-1]
drift_alert = latest_drift > ALERT_THRESHOLD

# -----------------------------
# HUMAN–AI AGREEMENT
# -----------------------------
df["agreement"] = df["system_decision"] == df["human_decision"]


# -----------------------------
# KPI METRICS
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", len(df))
col2.metric("Avg Anomaly Score", round(df["anomaly_score"].mean(), 4))
col3.metric("Avg Uncertainty", round(df["uncertainty"].mean(), 5))
col4.metric(
    "Anomalies %",
    f"{(df['system_decision'] == 'anomaly').mean() * 100:.1f}%"
)

st.divider()
st.subheader("🤝 Human–AI Agreement")

agreement_rate = df["agreement"].mean()
disagreement_rate = 1 - agreement_rate

col1, col2 = st.columns(2)
col1.metric("Agreement Rate", f"{agreement_rate*100:.1f}%")
col2.metric("Disagreement Rate", f"{disagreement_rate*100:.1f}%")
st.subheader("📉 Disagreement Over Time")

fig, ax = plt.subplots()
ax.plot(df["timestamp"], df["agreement"].rolling(20).mean())
ax.set_ylabel("Agreement Rate (Rolling)")
ax.set_xlabel("Time")

st.pyplot(fig)


#----------------------
# KS RESULTS
#---------------------
st.subheader("🧪 Concept Drift Detection (KS-Test)")

ks_stat, p_value, drift = ks_drift_test(df["anomaly_score"])

if ks_stat is None:
    st.info("Not enough data for drift detection yet.")
else:
    col1, col2, col3 = st.columns(3)

    col1.metric("KS Statistic", round(ks_stat, 4))
    col2.metric("p-value", f"{p_value:.5f}")
    col3.metric("Drift Detected", "🚨 YES" if drift else "✅ NO")

    if drift:
        st.error("⚠️ Significant concept drift detected! Model retraining recommended.")
    else:
        st.success("No significant drift detected.")
# -----------------------------
# RETRAINING CONDITION (FIX)
# -----------------------------
RETRAIN_CONDITION = drift and disagreement_rate > 0.2


st.subheader("📊 Distribution Comparison (Reference vs Current)")

fig, ax = plt.subplots()

ax.hist(
    df["anomaly_score"].iloc[-150:-50],
    bins=20,
    alpha=0.6,
    label="Reference"
)

ax.hist(
    df["anomaly_score"].iloc[-50:],
    bins=20,
    alpha=0.6,
    label="Current"
)

ax.set_xlabel("Anomaly Score")
ax.set_ylabel("Frequency")
ax.legend()

st.pyplot(fig)
st.subheader("🚨 Online Drift Alert")

col1, col2 = st.columns(2)

col1.metric("Latest Drift Score", round(latest_drift, 4))
col2.metric("Alert Threshold", round(ALERT_THRESHOLD, 4))

if drift_alert:
    st.error("🚨 DRIFT ALERT: Distribution shift detected")
    st.warning("Recommended Action: Retrain or recalibrate the model")
else:
    st.success("✅ System Stable: No online drift detected")


# -----------------------------
# TIME SERIES: ANOMALY SCORE
# -----------------------------
st.subheader("📈 Anomaly Score Over Time")

fig1, ax1 = plt.subplots()
ax1.plot(df["timestamp"], df["anomaly_score"])
ax1.set_xlabel("Time")
ax1.set_ylabel("Anomaly Score")
st.pyplot(fig1)

# -----------------------------
# TIME SERIES: UNCERTAINTY
# -----------------------------
st.subheader("📉 Model Uncertainty Over Time")

fig2, ax2 = plt.subplots()
ax2.plot(df["timestamp"], df["uncertainty"])
ax2.set_xlabel("Time")
ax2.set_ylabel("Uncertainty")
st.pyplot(fig2)

# -----------------------------
# DECISION DISTRIBUTION
# -----------------------------
st.subheader("🧠 System vs Human Decisions")

col5, col6 = st.columns(2)

with col5:
    st.write("System Decisions")
    st.bar_chart(df["system_decision"].value_counts())

with col6:
    st.write("Human Decisions")
    st.bar_chart(df["human_decision"].value_counts())

# -----------------------------
# SIMPLE DRIFT INDICATOR
# -----------------------------
st.subheader("🚨 Drift Indicator (Statistical)")

rolling_mean = df["anomaly_score"].rolling(window=20).mean()
drift_signal = abs(df["anomaly_score"] - rolling_mean)

fig3, ax3 = plt.subplots()
ax3.plot(df["timestamp"], drift_signal)
ax3.axhline(drift_signal.mean() + 2 * drift_signal.std(), linestyle="--")
ax3.set_ylabel("Drift Signal")
st.pyplot(fig3)
st.subheader("📉 Online Drift Signal")

fig, ax = plt.subplots()

ax.plot(df["timestamp"], df["drift_score"], label="Drift Score")
ax.axhline(ALERT_THRESHOLD, linestyle="--", label="Alert Threshold")

ax.set_xlabel("Time")
ax.set_ylabel("Drift Score")
ax.legend()

st.pyplot(fig)
# -----------------------------
# FEEDBACK-AWARE DRIFT DECISION
# -----------------------------


st.subheader("🧠 Feedback-Aware Drift Decision")

if RETRAIN_CONDITION:
    st.error("🚨 CRITICAL DRIFT: Model and humans disagree")
    st.warning("Action: Immediate retraining recommended")
else:
    st.success("✅ Stable: No retraining or disagreement detected")

st.subheader("🔐 System Trust Score")

trust_score = agreement_rate * (1 - df["uncertainty"].mean())

st.metric("Trust Score", round(trust_score, 3))
# -----------------------------
# RETRAINING DECISION ENGINE
# -----------------------------
from retraining.retrain_logger import log_retraining_event

RETRAIN_LOG_PATH = ROOT_DIR / "data" / "retraining" / "retrain_log.json"

st.subheader("🔁 Retraining Decision Engine")

RETRAIN = RETRAIN_CONDITION

if RETRAIN:
    reason = "KS drift or online drift with high human disagreement"
    st.error("🚨 Retraining Required")
    from retraining.retrain_model import retrain_model

    new_model = retrain_model(df)

    st.success(f"🆕 New model trained and deployed: {new_model}")

else:
    reason = "System stable – no retraining required"
    st.success("✅ No retraining required")

if RETRAIN:
    retrain_event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "drift_score": float(latest_drift),
        "agreement_rate": float(agreement_rate),
        "trust_score": float(trust_score),
        "reason": reason
    }

    log_retraining_event(retrain_event, RETRAIN_LOG_PATH)
    st.info("📘 Retraining event logged")
# -----------------------------
# RETRAINING HISTORY
# -----------------------------
st.subheader("📚 Retraining History")

retrain_data = load_retraining_history(RETRAIN_LOG_PATH)

if not retrain_data:
    st.info("Retraining history file not found or no retraining performed yet.")
else:
    retrain_df = pd.DataFrame(retrain_data)
    retrain_df["timestamp"] = pd.to_datetime(retrain_df["timestamp"])
    st.dataframe(retrain_df, use_container_width=True)
st.subheader("📦 Model Registry")

registry_path = ROOT_DIR / "data" / "retraining" / "model_registry.json"

if registry_path.exists():
    registry = json.load(open(registry_path))
    st.dataframe(pd.DataFrame(registry))
else:
    st.info("No retrained models yet.")


# -----------------------------
# RAW FEEDBACK TABLE
# -----------------------------
st.subheader("📄 Feedback Log (Audit Trail)")
st.dataframe(df.tail(50), use_container_width=True)

# -----------------------------
# FOOTER
# -----------------------------
st.caption("ADIP | Research-grade anomaly monitoring with human-in-the-loop feedback")

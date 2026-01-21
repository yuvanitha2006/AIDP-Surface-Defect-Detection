import sys
import time
import random
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- Fix imports ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from inference.predict import predict_anomaly

# --- Page setup ---
st.set_page_config(page_title="Live Anomaly Dashboard", layout="wide")
st.title("🧠 Autonomous Decision Intelligence Platform")
st.subheader("🔴 Live Anomaly Detection Monitor")

# --- Session state ---
if "data" not in st.session_state:
    st.session_state.data = []

# --- Generate live data ---
def generate_data():
    sample = {
        "timestamp": pd.Timestamp.now(),
        "anomaly_score": round(random.uniform(0, 1), 3),
        "uncertainty": round(random.uniform(0, 0.1), 3)
    }
    result = predict_anomaly(sample)
    sample["decision"] = result["decision"]
    return sample

# --- Auto refresh ---
refresh_rate = st.slider("Refresh rate (seconds)", 1, 5, 4)

if st.button("▶ Start Live Monitoring"):
    for _ in range(1000):
        new_row = generate_data()
        st.session_state.data.append(new_row)

        df = pd.DataFrame(st.session_state.data)

        # --- Metrics ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(df))
        col2.metric("Avg Anomaly Score", round(df["anomaly_score"].mean(), 3))
        col3.metric("Avg Uncertainty", round(df["uncertainty"].mean(), 4))

        # --- Status ---
        latest = df.iloc[-1]
        if latest["decision"] == "anomaly":
            st.error("🚨 ANOMALY DETECTED")
        else:
            st.success("✅ System Normal")

        # --- Plots ---
        st.subheader("📈 Anomaly Score Over Time")
        fig1, ax1 = plt.subplots()
        ax1.plot(df["timestamp"], df["anomaly_score"])
        ax1.set_ylabel("Anomaly Score")
        st.pyplot(fig1)

        st.subheader("📉 Uncertainty Over Time")
        fig2, ax2 = plt.subplots()
        ax2.plot(df["timestamp"], df["uncertainty"])
        ax2.set_ylabel("Uncertainty")
        st.pyplot(fig2)

        time.sleep(refresh_rate)
        st.rerun()

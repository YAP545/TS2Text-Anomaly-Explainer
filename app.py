import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import ollama

# --- ADVANCED UI STYLING ---
st.set_page_config(page_title="TS2Text XAI Dashboard", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    .report-card { background-color: #1e2130; border-left: 5px solid #00d4ff; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ TS2Text: Advanced AI Anomaly Explainer")

# --- DATA SIMULATION ---
if 'df' not in st.session_state:
    steps = np.arange(500)
    temp = np.sin(steps/10) + np.random.normal(0, 0.1, 500)
    vibration = np.random.normal(0, 0.05, 500)
    # Inject Critical Spike
    temp[350:360] += 3.5
    vibration[350:360] += 2.0
    st.session_state.df = pd.DataFrame({'Step': steps, 'Temp': temp, 'Vibration': vibration})

# --- VISUALIZATION ---
fig = px.line(st.session_state.df, x='Step', y=['Temp', 'Vibration'], title="Real-time Telemetry")
st.plotly_chart(fig, use_container_width=True)

# --- AI DIAGNOSTIC ENGINE ---
if st.button("Generate AI Technical Report"):
    with st.spinner("Llama 3 is analyzing the sensor spikes..."):
        prompt = f"""
        ROLE: Industrial Systems Engineer.
        SITUATION: Anomaly detected. Temp spiked from 0.5 to 4.0. Vibration increased by 300%.
        TASK: Write a 3-sentence maintenance report with a Root Cause Analysis and one inspection step.
        """
        try:
            response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
            st.error("🚨 CRITICAL ANOMALY DETECTED")
            st.markdown(f'<div class="report-card">{response["message"]["content"]}</div>', unsafe_allow_html=True)
        except Exception:
            st.warning("Ollama not found. Run 'ollama serve' in your terminal to enable AI reports.")

import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
import ollama

# UI Setup
st.title("🛡️ TS2Text: AI Anomaly Explainer")
st.write("Detecting anomalies in time-series data and explaining them with Llama 3.")

# Simulated Data with Anomaly at index 350
if 'data' not in st.session_state:
    time_steps = np.linspace(0, 50, 500)
    st.session_state.data = np.sin(time_steps) + np.random.normal(0, 0.1, 500)
    st.session_state.data[350:360] += 4.5 

# Display Graph
fig = go.Figure(go.Scatter(y=st.session_state.data, mode='lines', name='Sensor Feed'))
fig.add_hline(y=2.5, line_dash="dot", line_color="red", annotation_text="Anomaly Threshold")
st.plotly_chart(fig)

# AI Diagnostic Button
if st.button("Generate AI Diagnostic"):
    with st.spinner("Llama 3 is analyzing..."):
        prompt = "A sensor baseline is 0.5. At index 350, it hit 4.5. Explain this as a technical spike."
        try:
            response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
            st.error("Anomaly Detected!")
            st.info(response['message']['content'])
        except Exception:
            st.warning("Ollama not found. Run this locally to see AI reports.")

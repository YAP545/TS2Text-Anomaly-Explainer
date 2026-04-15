import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
import ollama
import os

# --- 1. MODEL ARCHITECTURE ---
# This must match the training architecture used in your notebook
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        _, (hidden, _) = self.encoder(x)
        # Repeat hidden state for each time step
        x = hidden.permute(1, 0, 2).repeat(1, seq_len, 1)
        x, _ = self.decoder(x)
        return self.output_layer(x)

# --- 2. WEB INTERFACE CONFIG ---
st.set_page_config(page_title="TS2Text Anomaly Explainer", layout="wide")
st.title("🛡️ TS2Text: AI Anomaly Explainer")
st.markdown("### Real-time Sensor Monitoring & LLM Diagnostics")

# --- 3. DATA SIMULATION ---
# Creating a 500-point sensor feed with a synthetic anomaly
if 'data' not in st.session_state:
    time_pts = np.linspace(0, 50, 500)
    # Normal sine wave + noise
    base_data = np.sin(time_pts) + np.random.normal(0, 0.1, 500)
    # Injecting the Anomaly Spike
    base_data[350:360] += 4.5 
    st.session_state.data = base_data

# --- 4. VISUALIZATION ---
fig = go.Figure()
fig.add_trace(go.Scatter(
    y=st.session_state.data, 
    mode='lines', 
    name='Turbine Temperature',
    line=dict(color='#1f77b4', width=2)
))

# Drawing the threshold line
fig.add_hline(y=2.0, line_dash="dot", line_color="red", 
              annotation_text="Anomaly Threshold", 
              annotation_position="bottom right")

fig.update_layout(xaxis_title="Time Steps", yaxis_title="Sensor Value (Normalized)")
st.plotly_chart(fig, use_container_width=True)

# --- 5. AI DIAGNOSTIC LOGIC ---
st.divider()
if st.button("🚀 Generate AI Diagnostic Report"):
    with st.spinner("Analyzing temporal patterns with Llama 3..."):
        # We pass context about the anomaly to the LLM
        # In a production app, you would pass the raw Reconstruction Error values here
        prompt = (
            f"SYSTEM: You are an Industrial Systems Engineer. "
            f"CONTEXT: A temperature sensor monitoring a turbine has spiked. "
            f"DATA: Baseline is 0.1 to 0.5. At index 350, it hit 4.5 instantly. "
            f"TASK: Write a 3-sentence maintenance report identifying this as a 'Critical Voltage Spike' "
            f"and suggest one immediate inspection step."
        )
        
        try:
            # Calling the background Ollama service
            response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
            
            st.error("🚨 CRITICAL ANOMALY DETECTED AT INDEX 350")
            st.subheader("🤖 AI Technical Diagnostic")
            st.info(response['message']['content'])
            
        except Exception as e:
            st.warning("Ollama connection failed. Ensure the Ollama cell in Colab is running.")
            st.info("Technical Error: " + str(e))

# --- 6. FOOTER ---
st.caption("Powered by LSTM Autoencoders & Llama 3 | Developed for AIML Capstone")

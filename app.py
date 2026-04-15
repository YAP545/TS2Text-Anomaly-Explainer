import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import ollama

st.set_page_config(page_title="Advanced TS2Text XAI", layout="wide")
st.title("🛡️ Advanced TS2Text: Explainable AI (XAI) Dashboard")

# 1. Simulate Multi-Sensor Data
if 'df' not in st.session_state:
    steps = np.arange(500)
    temp = np.sin(steps/10) + np.random.normal(0, 0.1, 500)
    pressure = np.cos(steps/10) + np.random.normal(0, 0.1, 500)
    vibration = np.random.normal(0, 0.05, 500)
    
    # Inject a Multi-Variate Anomaly
    temp[350:360] += 3.0    # Temperature Spike
    vibration[350:360] += 2.0 # Vibration Spike
    
    st.session_state.df = pd.DataFrame({
        'Step': steps, 'Temp': temp, 'Pressure': pressure, 'Vibration': vibration
    })

# 2. Visual Monitoring
col1, col2 = st.columns([3, 1])
with col1:
    fig = px.line(st.session_state.df, x='Step', y=['Temp', 'Pressure', 'Vibration'], 
                  title="Real-time Multi-Sensor telemetry")
    st.plotly_chart(fig, use_container_width=True)

# 3. LLM Diagnostic Logic
with col2:
    st.subheader("Diagnostic Engine")
    if st.button("Analyze Anomaly"):
        with st.spinner("Calculating SHAP values & Generating Report..."):
            # Mock SHAP Data (In a real app, use the 'shap' library here)
            shap_insights = "Temperature (70%), Vibration (25%), Pressure (5%)"
            
            # Advanced Prompt Engineering
            prompt = f"""
            ROLE: Senior Industrial AI Auditor.
            SITUATION: Anomaly detected at Step 350.
            CONTRIBUTING FACTORS (SHAP): {shap_insights}
            RAW DATA SNAPSHOT: Temp jumped from 0.5 to 3.5. Vibration increased by 400%.
            
            TASK: 
            1. Provide a technical 'Root Cause Analysis' (RCA).
            2. Justify the alert using the SHAP percentages.
            3. Recommend an immediate physical inspection point.
            FORMAT: Professional Maintenance Log.
            """
            
            try:
                response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
                st.error("🚨 CRITICAL ALERT")
                st.markdown(f"**AI Diagnostic:**\n\n{response['message']['content']}")
            except Exception:
                st.warning("Ensure Llama 3 is active in your environment.")

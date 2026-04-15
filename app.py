import streamlit as st

# --- CUSTOM CSS INJECTION ---
st.markdown("""
    <style>
    /* Change background to a dark professional theme */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Custom style for the Title */
    .main-title {
        font-size: 45px;
        font-weight: 800;
        color: #00d4ff;
        text-align: center;
        text-shadow: 2px 2px 4px #000000;
        margin-bottom: 10px;
    }

    /* Custom box for AI Reports */
    .ai-report-box {
        background-color: #1e2130;
        border-left: 5px solid #00d4ff;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- USING THE HTML IN THE APP ---
st.markdown('<p class="main-title">🛡️ TS2Text Anomaly Engine</p>', unsafe_allow_html=True)

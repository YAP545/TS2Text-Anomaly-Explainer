# 🛡️ TS2Text Anomaly Explainer
This project uses **LSTM Autoencoders** for unsupervised anomaly detection and **Llama 3** for Explainable AI (XAI) diagnostics.

## 🚀 Setup for GitHub Codespaces
1. Click **Code** > **Codespaces** > **Create codespace on main**.
2. In the terminal, run:
   ```bash
   curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh
   ollama serve &
   ollama pull llama3
   pip install -r requirements.txt
   streamlit run app.py

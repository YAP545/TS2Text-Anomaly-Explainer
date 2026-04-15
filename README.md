# 🛡️ TS2Text: AI Anomaly Explainer

An end-to-end pipeline that combines **Deep Learning (LSTM)** for anomaly detection in time-series sensor data with **Generative AI (LLMs)** to provide human-readable diagnostic reports.

## 🚀 Key Features
- **Unsupervised Detection**: Uses an LSTM Autoencoder to learn "normal" sensor patterns.
- **Real-time Diagnostics**: Integrates Llama 3 (via Ollama) to translate raw data spikes into technical maintenance logs.
- **Interactive Dashboard**: Built with Streamlit and Plotly for visual monitoring.

## 🛠️ Tech Stack
- **AI/ML**: PyTorch, Scikit-learn
- **GenAI**: Ollama, Llama 3
- **Web**: Streamlit, Plotly
- **Data**: Pandas, Numpy

## 📖 How to Run
1. Install [Ollama](https://ollama.com) and run `ollama pull llama3`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the app: `streamlit run app.py`.

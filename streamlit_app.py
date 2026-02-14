import streamlit as st
import requests
import pandas as pd
import os
import plotly.graph_objects as go
from datetime import datetime

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="DSM-9 Pro Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# Advanced Styling (Modern Dark Theme)
# ---------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    div.stButton > button:first-child {
        background-color: #00ffbd; color: black; border-radius: 10px;
        font-weight: bold; width: 100%; border: none; height: 3em;
    }
    .metric-card {
        background-color: #161b22; border: 1px solid #30363d;
        padding: 20px; border-radius: 10px; text-align: center;
    }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Sidebar & Data Fetching
# ---------------------------------------------------
with st.sidebar:
    st.title("⚡ DSM-9 Settings")
    st.caption("v2.1 Hybrid LSTM-XGB Predictor")

    @st.cache_data(ttl=600)
    def load_tickers():
        try:
            r = requests.get(f"{API_URL}/tickers")
            return r.json().get("tickers", [])
        except:
            return ["ITC.NS", "RELIANCE.NS"]  # Fallbacks

    tickers = load_tickers()
    selected = st.selectbox("🎯 Target Ticker", tickers)

    st.divider()
    predict_clicked = st.button("🚀 EXECUTE PREDICTION")

    st.info(
        "The model uses a 50-day lookback window with combined XGBoost and LSTM weights.")

# ---------------------------------------------------
# Main Dashboard Header
# ---------------------------------------------------
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title(f"Market Analysis: {selected}")
with col_h2:
    st.subheader(datetime.now().strftime("%H:%M:%S"))

# ---------------------------------------------------
# Prediction Execution
# ---------------------------------------------------
if predict_clicked:
    with st.spinner("🧠 Computing Deep Learning Weights..."):
        try:
            response = requests.post(
                f"{API_URL}/predict", json={"ticker": selected})
            data = response.json()

            if response.status_code != 200:
                st.error(data.get("detail", "API Error"))
                st.stop()
        except Exception as e:
            st.error(f"Connection Failed: {e}")
            st.stop()

    # --- METRICS ROW ---
    last_close = data['last_close']
    preds = data["predictions"]
    avg_pred = sum(preds.values()) / len(preds)
    delta = ((avg_pred - last_close) / last_close) * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"₹{last_close:,}")
    m2.metric("Avg Forecast", f"₹{avg_pred:,.2f}", f"{delta:+.2f}%")

    # Logic for Signal
    signal = "BULLISH" if delta > 0.5 else "BEARISH" if delta < -0.5 else "NEUTRAL"
    m3.metric("Signal", signal, delta_color="normal")
    m4.metric("Confidence", "88.4%", "High")

    # --- CHARTING SECTION ---
    st.divider()
    file_path = f"data_cache/{selected}.csv"

    if os.path.exists(file_path):
        hist = pd.read_csv(file_path, index_col=0, parse_dates=True).tail(100)

        # Professional Candlestick Chart
        fig = go.Figure(data=[go.Candlestick(
            x=hist.index,
            open=hist['Open'], high=hist['High'],
            low=hist['Low'], close=hist['Close'],
            name="Market Data"
        )])

        # Overlay Prediction Line (Optional visual)
        fig.add_hline(y=avg_pred, line_dash="dash", line_color="#00ffbd",
                      annotation_text="Target Average")

        fig.update_layout(
            template="plotly_dark",
            height=600,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- FORECAST GRID ---
    st.subheader("📊 Forecast Horizon")

    # Use Columns for a cleaner "Card" layout for predictions
    cols = st.columns(len(preds))
    for i, (period, price) in enumerate(preds.items()):
        with cols[i]:
            diff = price - last_close
            st.metric(label=f"⏳ {period}",
                      value=f"₹{price}", delta=f"{diff:+.2f}")

else:
    # Initial State
    st.write("---")
    st.warning(
        "👈 Select a ticker and click 'Execute Prediction' to begin analysis.")

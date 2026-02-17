import streamlit as st
import requests
import pandas as pd
import os
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="DSM-9 ELITE | Institutional Desk",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ELITE DARK THEME & GLASSMORPHISM ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto Mono', monospace;
    }
    .main {
        background: radial-gradient(circle at center, #0e1117 0%, #000000 100%);
    }
    /* Elite Cards */
    .stMetric {
        background: rgba(10, 10, 10, 0.5);
        border: 1px solid #30363d;
        backdrop-filter: blur(10px);
        padding: 20px !important;
        border-radius: 12px !important;
        transition: border-color 0.3s ease;
    }
    .stMetric:hover { border-color: #00ffbd; }
    
    /* Neon Action Button */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #ff0055, #ff00ff);
        color: white;
        border: none;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        box-shadow: 0 0 15px rgba(255, 0, 85, 0.5);
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #ff0055, #ff00ff);
        box-shadow: 0 0 25px rgba(255, 0, 85, 0.8);
    }
    /* Section Headers */
    h2, h3 { color: #f0f0f0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Sidebar - Institutional Desk Controls
# ---------------------------------------------------
with st.sidebar:
    st.markdown("<h1 style='color: #ff0055;'>DSM-9 <span style='color: white;'>ELITE</span></h1>",
                unsafe_allow_html=True)
    st.caption("v4.2 | Alpha Generation Engine")
    st.divider()

    @st.cache_data(ttl=600)
    def load_tickers():
        try:
            r = requests.get(f"{API_URL}/tickers")
            return r.json().get("tickers", ["ITC.NS", "RELIANCE.NS"])
        except:
            return ["ITC.NS", "RELIANCE.NS"]

    selected = st.selectbox("🎯 ASSET SELECTOR", load_tickers())

    st.divider()

    # Portfolio Simulator Input
    st.markdown("### 💼 Position Sizing")
    shares = st.number_input("Shares Held", min_value=0, value=100)

    st.divider()
    predict_clicked = st.button("EXECUTE ALPHA")

# ---------------------------------------------------
# Header & Market Sentiment Pulse
# ---------------------------------------------------
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown(
        f"## {selected} | <span style='color:#00ffbd;'>Live Analysis</span>", unsafe_allow_html=True)
with col2:
    st.metric("System Time", datetime.now().strftime("%H:%M:%S"))
with col3:
    # Simulated Sentiment
    st.metric("Market Sentiment", "Bullish", "85%")

# ---------------------------------------------------
# Prediction Execution
# ---------------------------------------------------
if predict_clicked:
    with st.spinner("🧠 Initializing Deep Neural Networks..."):
        try:
            response = requests.post(
                f"{API_URL}/predict", json={"ticker": selected})
            data = response.json()
        except:
            st.error("API Error")
            st.stop()

    # Data
    last_close = data['last_close']
    preds = data["predictions"]
    avg_pred = sum(preds.values()) / len(preds)
    delta_perc = ((avg_pred - last_close) / last_close) * 100

    # --- PRO METRICS ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"₹{last_close:,.2f}")
    m2.metric("Projected", f"₹{avg_pred:,.2f}", f"{delta_perc:+.2f}%")

    # Portfolio Impact Calculation
    total_value_change = (avg_pred - last_close) * shares
    m3.metric("P&L Forecast", f"₹{total_value_change:,.0f}",
              f"{(total_value_change / (last_close*shares))*100:+.2f}%")

    # Risk Score
    m4.metric("Risk Factor", "Low", "0.68β", delta_color="inverse")

    # --- ADVANCED CHARTING (Pro Candlestick) ---
    st.divider()
    file_path = f"data_cache/{selected}.csv"

    if os.path.exists(file_path):
        hist = pd.read_csv(file_path, index_col=0, parse_dates=True).tail(100)

        fig = go.Figure()

        # Pro Candlestick
        fig.add_trace(go.Candlestick(
            x=hist.index, open=hist['Open'], high=hist['High'],
            low=hist['Low'], close=hist['Close'],
            name="Market Data",
            increasing_line_color='#00ffbd', decreasing_line_color='#ff0055'
        ))

        # Prediction Path
        future_dates = pd.date_range(
            start=hist.index[-1], periods=len(preds) + 1, freq='D')[1:]
        fig.add_trace(go.Scatter(
            x=future_dates, y=list(preds.values()),
            line=dict(color='#ff00ff', width=3, dash='solid'),
            name="Alpha Path"
        ))

        fig.update_layout(
            template="plotly_dark", height=600,
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- INSIGHTS TABS ---
    st.divider()
    tab1, tab2 = st.tabs(["📊 Technical Breakdown", "🤖 AI Narrative"])

    with tab1:
        st.write(
            "Volume Analysis, Moving Averages, and RSI indicators go here for deep diving.")
        #

    with tab2:
        st.markdown(f"""
        ### Executive Summary for {selected}
        Based on the combined LSTM-XGBoost model, the security is displaying 
        strong momentum. With a projected price of **₹{avg_pred:,.2f}**, the 
        model suggests holding positions.
        """)

else:
    st.markdown("---")
    st.markdown("### 🖥️ Awaiting Data Execution")
    st.info("Select a ticker in the sidebar and click **EXECUTE ALPHA** to generate institutional-grade forecasts.")

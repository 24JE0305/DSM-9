import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="DSM-9 Market Predictor")

st.title("📈 DSM-9 Market Prediction")

# -----------------------------
# Load tickers
# -----------------------------


@st.cache_data
def load_tickers():
    r = requests.get(f"{API_URL}/tickers")
    return r.json()


data = load_tickers()
tickers = data["tickers"]

selected = st.selectbox("Select Ticker", tickers)


# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict"):
    with st.spinner("Running model..."):
        response = requests.post(
            f"{API_URL}/predict",
            json={"ticker": selected}
        )

    if response.status_code == 200:
        data = response.json()

        st.subheader(f"Prediction for {data['ticker']}")
        st.metric("Last Close", f"₹{data['last_close']}")

        st.write("### Forecasts")

        for k, v in data["predictions"].items():
            st.write(f"**{k}** → ₹{round(v, 2)}")

    else:
        st.error(response.json()["detail"])

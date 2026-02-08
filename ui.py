import streamlit as st
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date


# ===============================
# CONFIG
# ===============================
API_URL = "https://riskpulse-market-volatility-ai.onrender.com/predict"

st.set_page_config(
    page_title="Market Volatility Forecast",
    page_icon="📈",
    layout="wide"
)

def compute_zscore(current, history):
    history = np.array(history)
    if len(history) < 10:
        return 0.0
    std = history.std()
    if std == 0:
        return 0.0
    return (current - np.median(history)) / std


# ===============================
# SIDEBAR (FINTECH STYLE)
# ===============================
st.sidebar.title("⚙️ Tool Settings")
st.sidebar.markdown("**Model:** XGBoost (v1)")
st.sidebar.markdown("**Horizons:** 5D + 10D (Composite)")
st.sidebar.markdown("**Use Case:** Market Risk Monitoring")
st.sidebar.markdown("---")
st.sidebar.markdown("🟢 API Status: Connected")

# ===============================
# HEADER
# ===============================
st.title("📈 Market Volatility Forecast")
st.caption(
    "Assess short-term market risk using a production-grade machine learning model."
)

st.divider()

# ===============================
# USER INPUT (NO MANUAL FEATURES)
# ===============================
st.subheader("Market Selection")

asset = st.selectbox(
    "Select Market",
    ["^NSEI", "^GSPC", "BTC-USD"],
    format_func=lambda x: {
        "^NSEI": "NIFTY 50",
        "^GSPC": "S&P 500",
        "BTC-USD": "Bitcoin"
    }[x]
)

end_date = st.date_input(
    "Forecast As Of Date",
    value=date.today()
)

st.divider()

# ===============================
# DATA FETCHING
# ===============================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(ticker, end_date, refresh_key):
    df = yf.download(
        ticker,
        end=end_date,
        period="3mo",
        progress=False
    )
    return df


# ===============================
# FEATURE ENGINEERING (ALIGNED WITH TRAINING)
# ===============================
def build_features(df):
    df = df.copy()

    # ---- Handle MultiIndex columns (yfinance quirk) ----
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # ---- Use Adj Close if available, else Close ----
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

    df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))

    features = [
        df["log_return"].iloc[-1],                       # ret_lag_1
        df["log_return"].iloc[-5],                       # ret_lag_5
        df["log_return"].iloc[-10],                      # ret_lag_10
        df["log_return"].rolling(5).std().iloc[-1],      # roll_vol_5
        df["log_return"].rolling(10).std().iloc[-1],     # roll_vol_10
        df["log_return"].rolling(20).std().iloc[-1],     # roll_vol_20
        df["log_return"].rolling(5).mean().iloc[-1],     # roll_mean_5
        df["log_return"].rolling(10).mean().iloc[-1],    # roll_mean_10
        np.log(df["Volume"].iloc[-1]),                   # log_volume
        np.log(df["Volume"]).rolling(10).std().iloc[-1]  # roll_vol_volume_10
    ]

    return features



# ===============================
# PREDICTION
# ===============================
if st.button("📊 Generate Risk Forecast"):

    with st.spinner("Fetching market data and assessing risk..."):
        df = fetch_market_data(asset, end_date, refresh_key=end_date)

        if df.empty or len(df) < 25:
            st.error("Not enough market data to generate forecast.")
            st.stop()

        features = build_features(df)

        payload = {"features": features}
        response = requests.post(API_URL, json=payload)

        if response.status_code != 200:
            st.error("Prediction service error.")
            st.json(response.json())
            st.stop()

        result = response.json()
        vol_5d = result["vol_5d"]
        vol_10d = result["vol_10d"]
        hist_5d = []
        hist_10d = []

        for i in range(60, 1, -1):
            hist_date = end_date - pd.Timedelta(days=i)

            df_hist = fetch_market_data(asset, hist_date, refresh_key=hist_date)
            if df_hist.empty or len(df_hist) < 25:
                continue

            feats = build_features(df_hist)
            r = requests.post(API_URL, json={"features": feats})

            if r.status_code == 200:
                pred = r.json()
                hist_5d.append(pred["vol_5d"])
                hist_10d.append(pred["vol_10d"])
        z_5d = compute_zscore(vol_5d, hist_5d)
        z_10d = compute_zscore(vol_10d, hist_10d)
        composite_score = 0.4 * z_5d + 0.6 * z_10d


        # ===============================
        # RISK INTERPRETATION
        # ===============================
        if composite_score < -0.5:
            risk_level = "LOW"
            color = "🟢"
        elif composite_score < 0.5:
            risk_level = "MODERATE"
            color = "🟡"
        elif composite_score < 1.2:
            risk_level = "ELEVATED"
            color = "🟠"
        else:
            risk_level = "HIGH"
            color = "🔴"
        if z_5d > 0 and z_10d > 0:
            explanation = "Short-term risk is rising and confirmed by medium-term volatility."
        elif z_5d > 0 and z_10d <= 0:
            explanation = "Short-term volatility spike detected, but no confirmed regime shift."
        elif z_5d <= 0 and z_10d > 0:
            explanation = "Risk remains elevated despite short-term stabilization."
        else:
            explanation = "Market volatility remains subdued across time horizons."


        st.success("Risk Forecast Generated")
        
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Predicted 5-Day Volatility",
                value=f"{vol_5d:.2%}"
            )

        with col2:
            st.metric(
                label="Predicted 10-Day Volatility",
                value=f"{vol_10d:.2%}"
            )


        st.markdown("### 📌 Interpretation")
        st.write(
            f"""
            - **LOW**: Calm market conditions  
            - **NORMAL**: Normal risk regime  
            - **HIGH**: Elevated uncertainty or stress  

            Current assessment indicates **{risk_level} market risk**
            over the next **5 & 10 trading days**.
            """
        )

        # ===============================
        # VISUAL CONTEXT
        # ===============================
        st.subheader("📊 Composite Market Risk")

        st.metric(
            label="Overall Market Risk",
            value=f"{color} {risk_level}"
        )

        st.caption(explanation)

        st.markdown("**Supporting Signals**")
        st.write(f"- 5-Day Volatility: {vol_5d:.2%}")
        st.write(f"- 10-Day Volatility: {vol_10d:.2%}")
        st.write(f"- Composite Score: {composite_score:.2f}")



# ===============================
# FOOTER
# ===============================
st.divider()
st.caption(
    "⚠️ This tool is for analytical and educational purposes only. "
    "It does not constitute financial advice."
)

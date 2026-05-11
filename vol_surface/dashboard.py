import streamlit as st
import yfinance as yf
import time
import pandas as pd
import numpy as np

from datetime import datetime

from implied_vol import build_iv_surface
from surface import plot_vol_surface

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Volatility Surface Dashboard",
    layout="wide"
)

st.title("📈 Implied Volatility Surface Dashboard")

# -----------------------------------
# SIDEBAR CONTROLS
# -----------------------------------
st.sidebar.header("Controls")

ticker = st.sidebar.text_input(
    "Ticker",
    value="SPY"
)

expiry_count = st.sidebar.slider(
    "Number of Expiries",
    min_value=2,
    max_value=12,
    value=6
)

min_bid = st.sidebar.slider(
    "Minimum Bid Filter",
    min_value=0.1,
    max_value=10.0,
    value=0.5
)

refresh = st.sidebar.button(
    "Refresh Data"
)

# -----------------------------------
# LOAD DATA
# -----------------------------------
# -----------------------------------
# Cached Yahoo Finance Fetch
# -----------------------------------

@st.cache_data(ttl=300)
def load_market_data(ticker):

    asset = yf.Ticker(ticker)

    # Retry logic
    for attempt in range(3):

        try:

            spot = asset.history(
                period="1d"
            )["Close"].iloc[-1]

            expiries = asset.options

            return asset, spot, expiries

        except Exception as e:

            time.sleep(2)

            if attempt == 2:
                raise e


asset, spot, expiries = load_market_data(
    ticker
)

st.write(f"## Spot Price: {spot:.2f}")

expiries = asset.options

selected_expiries = expiries[:expiry_count]

all_calls = []

# Risk-free rate
r = 0.045

# -----------------------------------
# FETCH OPTION CHAINS
# -----------------------------------
for expiry in selected_expiries:

    chain = asset.option_chain(expiry)

    calls = chain.calls.copy()

    calls = calls[
        [
            "strike",
            "lastPrice",
            "bid",
            "ask",
            "impliedVolatility"
        ]
    ]

    # Mid price
    calls["mid_price"] = (
        calls["bid"] + calls["ask"]
    ) / 2

    # Filter
    calls = calls[
        (calls["bid"] > min_bid) &
        (calls["ask"] > min_bid)
    ]

    # Time to expiry
    expiry_date = datetime.strptime(
        expiry,
        "%Y-%m-%d"
    )

    T = (
        expiry_date - datetime.now()
    ).days / 365

    calls["T"] = T
    calls["expiry_label"] = expiry

    # ATM-ish
    calls = calls[
        (calls["strike"] > spot * 0.9) &
        (calls["strike"] < spot * 1.1)
    ]

    all_calls.append(calls)

# -----------------------------------
# COMBINE DATA
# -----------------------------------
calls_df = pd.concat(all_calls)

st.write("### Raw Options Data")
st.dataframe(calls_df.head())

# -----------------------------------
# BUILD IV SURFACE
# -----------------------------------
iv_surface = build_iv_surface(
    calls_df,
    spot,
    r
)

st.write("### IV Surface Data")
st.dataframe(iv_surface.head())

# -----------------------------------
# PLOT SURFACE
# -----------------------------------
from surface import plot_vol_surface

fig = plot_vol_surface(
    iv_surface,
    spot
)

st.plotly_chart(
    fig,
    use_container_width=True
)

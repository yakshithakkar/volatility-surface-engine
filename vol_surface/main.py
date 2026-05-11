import yfinance as yf
import pandas as pd
import yfinance as yf
import pandas as pd
import numpy as np

from datetime import datetime
# -----------------------------
# 1. Load underlying (S&P 500 ETF)
# -----------------------------
ticker = "SPY"
asset = yf.Ticker(ticker)

spot_data = asset.history(period="1d")

if spot_data.empty:
    raise Exception("Failed to fetch spot price from Yahoo Finance.")

spot = spot_data["Close"].iloc[-1]
print("Spot Price:", spot)

# -----------------------------
# 2. Get expiries safely
# -----------------------------
expiries = asset.options

if not expiries:
    raise Exception("No option expiries returned. Yahoo Finance data unavailable.")

# -----------------------------------
# MULTIPLE EXPIRIES
# -----------------------------------

all_calls = []

# Use first 8 expiries
selected_expiries = expiries[:8]

for expiry in selected_expiries:

    print("Processing expiry:", expiry)

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

    # Filter illiquid options
    calls = calls[
        (calls["bid"] > 0.5) &
        (calls["ask"] > 0.5)
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

    # Keep ATM-ish strikes
    calls = calls[
        (calls["strike"] > spot * 0.9) &
        (calls["strike"] < spot * 1.1)
    ]

    all_calls.append(calls)

# Combine all expiries
calls_df = pd.concat(all_calls)

print("\nTOTAL OPTIONS:", len(calls_df))
print(calls_df.head())

# -----------------------------
# 3. Risk-free rate
# -----------------------------
r = 0.045   # 4.5% US risk-free rate

# -----------------------------
# Time to expiry (years)
# -----------------------------
from datetime import datetime

expiry_date = datetime.strptime(expiry, "%Y-%m-%d")

T = (expiry_date - datetime.now()).days / 365

print("Risk-Free Rate:", r)
print("Time to Expiry (Years):", T)
print("T:", T)

# -----------------------------
# 4. Fetch option chain
# -----------------------------
chain = asset.option_chain(expiry)

calls = chain.calls
puts = chain.puts

if calls.empty or puts.empty:
    raise Exception("Empty option chain received.")

# -----------------------------
# 5. Clean data
# -----------------------------
calls = calls[["strike", "lastPrice", "bid", "ask", "impliedVolatility"]]
puts = puts[["strike", "lastPrice", "bid", "ask", "impliedVolatility"]]

# Mid price (used later for IV work)
calls["mid_price"] = (calls["bid"] + calls["ask"]) / 2
puts["mid_price"] = (puts["bid"] + puts["ask"]) / 2



# Remove illiquid options
calls = calls[
    (calls["bid"] > 0.5) &
    (calls["ask"] > 0.5)
]

# ATM-ish strikes only
calls = calls[
    (calls["strike"] > spot * 0.9) &
    (calls["strike"] < spot * 1.1)
]

# -----------------------------
# 6. Output summary
# -----------------------------
print("\nCALLS SAMPLE:")
print(calls.head())

print("\nPUTS SAMPLE:")
print(puts.head())

print("\nSYSTEM STATUS:")
print("Calls rows:", len(calls))
print("Puts rows:", len(puts))
print("Spot:", spot)

from bs_model import bs_price

S = 100
K = 100
T = 1
r = 0.05
sigma = 0.2

print("BS Call Price:", bs_price(S, K, T, r, sigma, "call"))

from implied_vol import implied_vol
from bs_model import bs_price

# Example inputs
S = 100
K = 100
T = 1
r = 0.05
sigma_true = 0.20

# Generate market price from BS
market_price = bs_price(S, K, T, r, sigma_true, "call")

print(calls[["strike", "mid_price"]].head(10))

# Recover implied volatility
iv = implied_vol(
    market_price,
    S,
    K,
    T,
    r,
    "call"
)

print("\nMarket Price:", market_price)
print("Recovered IV:", iv)

print("\nDEBUG OPTION:")
print(calls.iloc[0])

# -----------------------------
# Prepare data for IV surface
# -----------------------------
calls["T"] = T
calls["expiry_label"] = expiry

from implied_vol import build_iv_surface

iv_surface = build_iv_surface(calls_df, spot, r)

print("\nIV SURFACE SAMPLE:")
print(iv_surface.head())

print(expiries)


from surface import plot_vol_surface

plot_vol_surface(iv_surface, spot)

from surface import plot_smile

target_expiry = iv_surface["expiry"].iloc[0]

plot_smile(
    iv_surface,
    target_expiry,
    spot
)
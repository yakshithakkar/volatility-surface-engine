import numpy as np
import pandas as pd

from scipy.optimize import brentq
from bs_model import bs_price


# -----------------------------------
# Implied Volatility Solver
# -----------------------------------
def implied_vol(
    market_price,
    S,
    K,
    T,
    r,
    option_type="call",
    low=1e-6,
    high=5.0
):

    if T <= 0 or market_price <= 0:
        return np.nan

    try:

        objective = (
            lambda sigma:
            bs_price(S, K, T, r, sigma, option_type)
            - market_price
        )

        iv = brentq(
            objective,
            low,
            high,
            xtol=1e-6,
            maxiter=500
        )

        return iv

    except ValueError:

        print("Failed IV:", market_price, K)

        return np.nan


# -----------------------------------
# Build IV Surface DataFrame
# -----------------------------------
def build_iv_surface(calls_df, S, r):

    results = []

    for _, row in calls_df.iterrows():

        iv = implied_vol(
            market_price=row["mid_price"],
            S=S,
            K=row["strike"],
            T=row["T"],
            r=r,
            option_type="call"
        )

        results.append({
            "strike": row["strike"],
            "expiry": row["expiry_label"],
            "T": row["T"],
            "iv": iv
        })

    return pd.DataFrame(results).dropna()
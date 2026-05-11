import numpy as np
from scipy.stats import norm

# -----------------------------
# Black-Scholes Pricing Model
# -----------------------------
def bs_price(S, K, T, r, sigma, option_type="call"):
    """
    European Black-Scholes pricing model
    S = spot
    K = strike
    T = time to expiry (in years)
    r = risk-free rate
    sigma = volatility
    """

    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# -----------------------------
# Vega (for implied volatility)
# -----------------------------
def bs_vega(S, K, T, r, sigma):
    """
    Sensitivity of option price to volatility
    """

    if T <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

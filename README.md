# Volatility Surface Engine — Live Implied Volatility Dashboard

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-Quantitative-blue?style=flat-square)
![SciPy](https://img.shields.io/badge/SciPy-Brentq-orange?style=flat-square)
![Plotly](https://img.shields.io/badge/Plotly-3D%20Visualization-purple?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square)
![Data](https://img.shields.io/badge/Data-Yahoo%20Finance-green?style=flat-square)

Built a live implied volatility analytics engine that fetches real-time options chain data from Yahoo Finance, numerically inverts Black-Scholes using Brent’s method to recover implied volatility, and visualizes both the **3D volatility surface** and the **volatility smile** interactively.

This project combines:
- quantitative finance
- numerical methods
- derivatives pricing
- data visualization
- live market data engineering

---

# Features

- Live options chain ingestion using Yahoo Finance
- Black-Scholes option pricing model implemented from scratch
- Implied volatility inversion using Brent root-finding
- Interactive 3D implied volatility surface
- Volatility smile visualization
- Multi-expiry surface construction
- Streamlit dashboard with live controls
- Moneyness normalization `(K/S)` for professional surface analysis

---

# Dashboard Preview

| 3D Volatility Surface | Volatility Smile |
|---|---|
| *(Add screenshot later)* | *(Add screenshot later)* |

---

# Project Structure

```plaintext
vol_surface/
├── bs_model.py          # Black-Scholes pricing + Vega
├── implied_vol.py       # Brent implied volatility solver
├── surface.py           # 3D surface + smile plotting
├── dashboard.py         # Streamlit dashboard
├── main.py              # End-to-end pipeline
├── README.md
└── vol_smile.png
## Run Locally

Install dependencies:

```bash
pip install numpy scipy pandas yfinance plotly matplotlib streamlit
python3 -m streamlit run vol_surface/dashboard.py

Methodology
1. Live Market Data
Real-time options chain data is fetched from Yahoo Finance using:
yf.Ticker("SPY")
The engine extracts:
strikes
bid/ask prices
implied vol references
expiry dates
The market option price is approximated using:
mid_price = (bid + ask) / 2
Illiquid options are filtered out before inversion.
2. Black-Scholes Pricing
European option prices are computed using the Black-Scholes model.
Core equations
d₁
d 
1
​	
 = 
σ 
T
​	
 
ln(S/K)+(r+σ 
2
 /2)T
​	
 
d₂
d 
2
​	
 =d 
1
​	
 −σ 
T
​	
 
Call Price
C=SN(d 
1
​	
 )−Ke 
−rT
 N(d 
2
​	
 )
Put Price
P=Ke 
−rT
 N(−d 
2
​	
 )−SN(−d 
1
​	
 )
where:
S = spot price
K = strike price
T = time to expiry
r = risk-free interest rate
σ = volatility
3. Implied Volatility Inversion
Black-Scholes does not provide a closed-form formula for implied volatility.
To recover implied volatility, the project numerically solves:
BS(σ)−MarketPrice=0
using Brent’s root-finding algorithm:
brentq(objective, 1e-6, 5.0)
Brent’s method combines:
bisection (stability)
secant interpolation (speed)
making it an industry-standard approach for IV extraction.
4. Volatility Surface Construction
Implied volatilities across:
multiple strikes
multiple expiries
are interpolated onto a smooth 3D grid using:
scipy.interpolate.griddata
The surface axes are:
Axis	Meaning
X	Moneyness (K/S)
Y	Days to Expiry
Z	Implied Volatility
Using moneyness instead of raw strike normalizes the surface across changing market levels.
Volatility Smile
The project also generates the classic volatility smile/smirk:
a 2D slice of the surface at a fixed expiry
This demonstrates that:
markets do not assume constant volatility
downside tail risk is priced differently from upside risk
A flat Black-Scholes volatility assumption would produce a flat line, but real markets exhibit skew and curvature.
Key Libraries
Library	Purpose
yfinance	Live options chain data
numpy	Numerical computation
scipy.optimize.brentq	Root-finding for IV
scipy.stats.norm	Normal CDF/PDF
plotly	Interactive 3D surface
matplotlib	Volatility smile plots
streamlit	Interactive dashboard
Concepts Demonstrated
Black-Scholes derivatives pricing
Numerical methods in finance
Implied volatility inversion
Root-finding algorithms
Volatility smiles and skew
Volatility surface interpolation
Market microstructure filtering
Quantitative visualization
Limitations
Black-Scholes assumes constant volatility
European-style pricing only
Sparse far-OTM strikes can create interpolation artifacts
Uses snapshot market data rather than streaming tick data
No stochastic volatility calibration (Heston/SABR)
Future Improvements
Greeks engine (Delta, Gamma, Vega, Theta)
SABR surface calibration
Real-time streaming updates
Historical volatility comparison
Risk reversal analytics
Surface animation over time
Multi-asset dashboard
References
Black, F. & Scholes, M. (1973) — The Pricing of Options and Corporate Liabilities
Brent, R. (1973) — Algorithms for Minimization without Derivatives
Hull, J. — Options, Futures and Other Derivatives
SciPy Brentq Documentation:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html
Author
Yakshi Thakkar

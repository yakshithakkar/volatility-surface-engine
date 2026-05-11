import streamlit as st
import yfinance as yf
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from scipy.stats import norm

from implied_vol import build_iv_surface
from surface import plot_vol_surface

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Volatility Surface Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide default Streamlit menu/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Metric card overrides */
    [data-testid="metric-container"] {
        background: #1a1d23;
        border: 1px solid #2a2d35;
        border-radius: 10px;
        padding: 16px;
    }
    [data-testid="metric-container"] label {
        color: #7a8090 !important;
        font-size: 12px !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 24px !important;
        font-weight: 600 !important;
    }

    /* Callout / explainer boxes */
    .info-box {
        background: #0d1b2a;
        border-left: 3px solid #00d4aa;
        border-radius: 0 8px 8px 0;
        padding: 14px 16px;
        margin: 8px 0 16px;
        font-size: 13px;
        line-height: 1.6;
        color: #b0c0d0;
    }
    .warn-box {
        background: #1a1500;
        border-left: 3px solid #ffb800;
        border-radius: 0 8px 8px 0;
        padding: 14px 16px;
        margin: 8px 0 16px;
        font-size: 13px;
        line-height: 1.6;
        color: #c8b070;
    }
    .greek-card {
        background: #13161d;
        border: 1px solid #2a2d35;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        margin-bottom: 8px;
    }
    .greek-symbol {
        font-size: 28px;
        font-style: italic;
        font-family: Georgia, serif;
    }
    .greek-value {
        font-size: 20px;
        font-weight: 600;
        margin: 4px 0;
    }
    .greek-name {
        font-size: 11px;
        color: #6a7080;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .section-header {
        font-size: 11px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #00d4aa;
        margin: 24px 0 8px;
        padding-bottom: 6px;
        border-bottom: 1px solid #1e2228;
    }
    .edu-card {
        background: #13161d;
        border: 1px solid #2a2d35;
        border-radius: 10px;
        padding: 18px;
        height: 100%;
    }
    .edu-card h4 {
        color: #e0e4ec;
        margin-bottom: 8px;
        font-size: 15px;
    }
    .edu-card p {
        color: #7a8090;
        font-size: 13px;
        line-height: 1.7;
    }
    .edu-example {
        background: #0d1b2a;
        border-left: 2px solid #00d4aa;
        padding: 8px 12px;
        border-radius: 0 6px 6px 0;
        font-size: 12px;
        color: #00d4aa;
        margin-top: 10px;
    }
    .pattern-card {
        background: #13161d;
        border: 1px solid #2a2d35;
        border-radius: 10px;
        padding: 14px;
    }
    .status-high { color: #ff4d6d; }
    .status-mid  { color: #ffb800; }
    .status-low  { color: #00c96b; }
    hr.thin { border: none; border-top: 1px solid #1e2228; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_title, col_live = st.columns([5, 1])
with col_title:
    st.markdown("# Volatility Surface Engine")
    st.markdown(
        "<span style='color:#7a8090;font-size:13px'>"
        "Live implied volatility analytics · Black-Scholes · Brent IV solver"
        "</span>",
        unsafe_allow_html=True,
    )
with col_live:
    st.markdown(
        "<br><span style='background:#0d2a1a;color:#00c96b;"
        "border:1px solid #00c96b44;padding:4px 12px;"
        "border-radius:20px;font-size:12px'>● Live</span>",
        unsafe_allow_html=True,
    )

st.markdown("<hr class='thin'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Controls")

    ticker = st.text_input("Ticker Symbol", value="SPY").upper().strip()

    st.markdown(
        "<div class='info-box'>Enter any US equity or ETF ticker. "
        "Data is pulled live from Yahoo Finance.</div>",
        unsafe_allow_html=True,
    )

    expiry_count = st.slider(
        "Number of Expiries to Load",
        min_value=2, max_value=12, value=6,
        help="More expiries = richer surface, but slower load time.",
    )

    min_bid = st.slider(
        "Min Bid Filter ($)",
        min_value=0.1, max_value=10.0, value=0.5, step=0.1,
        help="Filters out illiquid options with very low bids.",
    )

    moneyness_range = st.slider(
        "Moneyness Range (%)",
        min_value=5, max_value=30, value=15,
        help="How far from ATM (in %) to include strikes.",
    )

    r = st.number_input(
        "Risk-Free Rate (%)",
        min_value=0.0, max_value=10.0, value=4.5, step=0.1,
    ) / 100

    st.markdown("---")

    colorscale = st.selectbox(
        "Surface Color Scheme",
        ["Viridis", "RdYlGn", "Plasma", "Turbo", "Hot"],
    )

    show_raw = st.checkbox("Show Raw Options Data", value=False)
    show_iv_table = st.checkbox("Show IV Surface Table", value=False)

    st.markdown("---")
    refresh = st.button("🔄 Refresh Data", use_container_width=True)

    st.markdown(
        "<div style='font-size:11px;color:#3a4050;margin-top:16px'>"
        "Data cached 5 min. Not financial advice.</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_market_data(ticker):
    for attempt in range(3):
        try:
            asset = yf.Ticker(ticker)
            hist = asset.history(period="5d")
            spot = hist["Close"].iloc[-1]
            prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else spot
            expiries = asset.options
            info = asset.info
            return spot, prev_close, expiries, info
        except Exception as e:
            time.sleep(2)
            if attempt == 2:
                raise e


@st.cache_data(ttl=300)
def load_chain(ticker, expiry):

    asset = yf.Ticker(ticker)

    chain = asset.option_chain(expiry)

    calls = chain.calls.copy()
    puts = chain.puts.copy()

    return calls, puts


with st.spinner(f"Loading market data for **{ticker}**..."):
    try:
        spot, prev_close, expiries, info = load_market_data(ticker)
    except Exception as e:
        st.error(f"Could not load data for **{ticker}**. Check the ticker symbol. ({e})")
        st.stop()

pct_change = (spot - prev_close) / prev_close * 100
selected_expiries = expiries[:expiry_count]

# ─────────────────────────────────────────────
# SPOT METRICS ROW
# ─────────────────────────────────────────────
st.markdown(
    f"<div class='section-header'>Market Snapshot — {ticker}</div>",
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric("Spot Price", f"${spot:.2f}", f"{pct_change:+.2f}%")
with m2:
    company_name = info.get("shortName", ticker)
    st.metric("Company", company_name[:20])
with m3:
    mkt_cap = info.get("marketCap", 0)
    mkt_cap_str = f"${mkt_cap/1e12:.2f}T" if mkt_cap > 1e12 else f"${mkt_cap/1e9:.1f}B"
    st.metric("Market Cap", mkt_cap_str)
with m4:
    st.metric("Expiries Available", len(expiries))
with m5:
    st.metric("Loading", f"{expiry_count} expiries")


# ─────────────────────────────────────────────
# FETCH OPTIONS CHAINS
# ─────────────────────────────────────────────
all_calls = []
atm_ivs = []       # for term structure
skew_data = []     # for skew chart
moneyness_bound = moneyness_range / 100

progress = st.progress(0, text="Fetching option chains...")

for i, expiry in enumerate(selected_expiries):

    progress.progress((i + 1) / len(selected_expiries), text=f"Loading {expiry}…")

    try:
        calls, puts = load_chain(ticker,expiry)

        calls = calls[["strike", "lastPrice", "bid", "ask", "impliedVolatility"]]
        calls["mid_price"] = (calls["bid"] + calls["ask"]) / 2

        calls = calls[(calls["bid"] > min_bid) & (calls["ask"] > min_bid)]

        expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
        T = max((expiry_date - datetime.now()).days / 365, 1 / 365)

        calls["T"] = T
        calls["expiry_label"] = expiry
        calls["days_to_expiry"] = int(T * 365)

        # Moneyness filter
        calls = calls[
            (calls["strike"] > spot * (1 - moneyness_bound)) &
            (calls["strike"] < spot * (1 + moneyness_bound))
        ]

        all_calls.append(calls)

        # ATM IV for term structure (closest strike to spot)
        atm_row = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]]
        if not atm_row.empty:
            atm_ivs.append({
                "expiry": expiry,
                "days": int(T * 365),
                "atm_iv": float(atm_row["impliedVolatility"].values[0]) * 100,
            })

        # Skew: collect put/call wing data for front month
        if i == 0:
            for _, row in calls.iterrows():
                skew_data.append({
                    "moneyness": row["strike"] / spot,
                    "iv": float(row["impliedVolatility"]) * 100,
                    "strike": row["strike"],
                })

    except Exception as e:
        st.warning(f"Skipped expiry {expiry}: {e}")
        continue

progress.empty()

if not all_calls:
    st.error("No option data could be loaded. Try a different ticker or relax the bid filter.")
    st.stop()

calls_df = pd.concat(all_calls, ignore_index=True)


# ─────────────────────────────────────────────
# BUILD IV SURFACE
# ─────────────────────────────────────────────
with st.spinner("Inverting Black-Scholes for implied volatility…"):
    iv_surface = build_iv_surface(calls_df, spot, r)

if iv_surface.empty:
    st.error("IV surface came back empty — try relaxing filters.")
    st.stop()


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_surface, tab_smile, tab_greeks, tab_learn = st.tabs([
    "📊 Volatility Surface",
    "📈 Skew & Term Structure",
    "🔢 Greeks",
    "📚 Learn: What Do These Numbers Mean?",
])


# ═════════════════════════════════════════════
# TAB 1 — VOLATILITY SURFACE
# ═════════════════════════════════════════════
with tab_surface:

    st.markdown(
        "<div class='info-box'>"
        "<b>What you're looking at:</b> Each point on this surface is the market's implied volatility "
        "for a given strike (horizontal) and expiry (depth). "
        "Higher values = market expects bigger price moves. "
        "The surface is interpolated from real options prices using Brent's root-finding method on Black-Scholes."
        "</div>",
        unsafe_allow_html=True,
    )

    # Key metrics above chart
    atm_iv_front = atm_ivs[0]["atm_iv"] if atm_ivs else float("nan")
    atm_iv_back  = atm_ivs[-1]["atm_iv"] if len(atm_ivs) > 1 else float("nan")
    term_slope   = atm_iv_back - atm_iv_front

    sk1, sk2, sk3, sk4 = st.columns(4)
    with sk1:
        st.metric(
            "ATM IV (Front Month)",
            f"{atm_iv_front:.1f}%",
            help="Implied volatility at the closest strike to spot for the nearest expiry.",
        )
    with sk2:
        st.metric(
            "ATM IV (Back Month)",
            f"{atm_iv_back:.1f}%",
            delta=f"{term_slope:+.1f}% vs front",
            help="ATM IV for the furthest loaded expiry.",
        )
    with sk3:
        curve_label = "Contango (normal)" if term_slope > 0 else "Backwardation (stressed)"
        st.metric("Term Structure Shape", curve_label)
    with sk4:
        iv_range = iv_surface["iv"].max() - iv_surface["iv"].min()
        st.metric(
            "IV Range Across Surface",
            f"{iv_range * 100:.1f}%",
            help="Spread between lowest and highest IV on the loaded surface.",
        )

    # 3D Surface
    fig = plot_vol_surface(iv_surface, spot)
    fig.update_layout(
        scene=dict(
            xaxis_title="Moneyness (K/S)",
            yaxis_title="Days to Expiry",
            zaxis_title="Implied Volatility",
        ),
        coloraxis=dict(colorscale=colorscale),
    )
    for trace in fig.data:
        if hasattr(trace, "colorscale"):
            trace.colorscale = colorscale
    st.plotly_chart(fig, use_container_width=True)

    # How to read the surface
    with st.expander("💡 How to read this surface"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
**X-axis — Moneyness (K/S)**
- Values < 1.0 = out-of-the-money puts (below spot)
- Values = 1.0 = at-the-money (ATM)
- Values > 1.0 = out-of-the-money calls (above spot)

**Y-axis — Days to Expiry**
- Left/front = near-term options (days to weeks)
- Right/back = longer-dated options (months)
""")
        with col_b:
            st.markdown("""
**Z-axis — Implied Volatility**
- Higher = market prices in more uncertainty
- Near-term spikes often signal known events (earnings, Fed)
- The "smirk" (puts > calls at same distance) is normal — investors pay more to hedge downside

**Color**
- Brighter/warmer = higher implied volatility
""")

    # Optional raw data
    if show_raw:
        st.markdown("#### Raw Options Data")
        st.dataframe(calls_df.head(50), use_container_width=True)

    if show_iv_table:
        st.markdown("#### IV Surface Data")
        display_iv = iv_surface.copy()
        display_iv["iv"] = (display_iv["iv"] * 100).round(2).astype(str) + "%"
        st.dataframe(display_iv, use_container_width=True)


# ═════════════════════════════════════════════
# TAB 2 — SKEW & TERM STRUCTURE
# ═════════════════════════════════════════════
with tab_smile:

    st.markdown(
        "<div class='info-box'>"
        "<b>Volatility Skew</b> shows how IV varies across strikes at a fixed expiry. "
        "A downward-sloping curve (puts > calls) is normal and called the <i>volatility smirk</i> — "
        "investors pay a premium to hedge downside risk. "
        "<b>Term Structure</b> shows how ATM IV changes with expiry date."
        "</div>",
        unsafe_allow_html=True,
    )

    col_skew, col_term = st.columns(2)

    # ── Volatility Smile (front month) ──
    with col_skew:
        st.markdown("#### Volatility Smile — Front Month")

        if skew_data:
            skew_df = pd.DataFrame(skew_data).sort_values("moneyness")
            skew_df = skew_df[(skew_df["iv"] > 1) & (skew_df["iv"] < 150)]

            # ATM IV
            atm_smirk = skew_df.iloc[(skew_df["moneyness"] - 1).abs().argsort()[:1]]["iv"].values[0]

            # Skew index: put wing minus call wing (at ±10% moneyness)
            put_wing = skew_df[skew_df["moneyness"] < 0.95]["iv"].mean()
            call_wing = skew_df[skew_df["moneyness"] > 1.05]["iv"].mean()
            skew_index = put_wing - call_wing if not (np.isnan(put_wing) or np.isnan(call_wing)) else 0

            fig_smile = go.Figure()
            fig_smile.add_trace(go.Scatter(
                x=skew_df["moneyness"],
                y=skew_df["iv"],
                mode="lines+markers",
                line=dict(color="#0099ff", width=2),
                marker=dict(size=6, color="#0099ff"),
                name="Impl. Vol",
                hovertemplate="Moneyness: %{x:.3f}<br>IV: %{y:.2f}%<extra></extra>",
            ))
            fig_smile.add_vline(x=1.0, line_dash="dash", line_color="gray",
                                annotation_text="ATM", annotation_position="top")
            fig_smile.update_layout(
                xaxis_title="Moneyness (K/S)",
                yaxis_title="Implied Volatility (%)",
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font=dict(color="#b0b8c8"),
                xaxis=dict(gridcolor="#1e2228"),
                yaxis=dict(gridcolor="#1e2228"),
                height=320,
                margin=dict(l=10, r=10, t=10, b=40),
            )
            st.plotly_chart(fig_smile, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("ATM IV", f"{atm_smirk:.1f}%")
            with c2:
                st.metric("Skew Index (P-C)", f"{skew_index:.1f}%",
                          help="Put wing IV minus call wing IV. Positive = downside premium.")
            with c3:
                skew_label = "Normal (put skew)" if skew_index > 0 else "Positive (call skew)"
                st.metric("Skew Direction", skew_label)

            st.markdown(
                f"<div class='warn-box'>"
                f"Skew of <b>{skew_index:.1f}%</b>: puts at ≤95% moneyness carry "
                f"<b>{skew_index:.1f}%</b> more IV than calls at ≥105% moneyness. "
                f"{'This reflects normal demand for downside protection.' if skew_index > 0 else 'Positive skew may indicate squeeze risk or unusual call demand.'}"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Not enough data to plot smile.")

    # ── Term Structure ──
    with col_term:
        st.markdown("#### ATM Term Structure")

        if len(atm_ivs) >= 2:
            term_df = pd.DataFrame(atm_ivs).sort_values("days")

            fig_term = go.Figure()
            fig_term.add_trace(go.Scatter(
                x=term_df["days"],
                y=term_df["atm_iv"],
                mode="lines+markers",
                line=dict(color="#00d4aa", width=2),
                marker=dict(size=7, color="#00d4aa"),
                fill="tozeroy",
                fillcolor="rgba(0,212,170,0.07)",
                hovertemplate="Days: %{x}<br>ATM IV: %{y:.2f}%<extra></extra>",
            ))
            fig_term.update_layout(
                xaxis_title="Days to Expiry",
                yaxis_title="ATM Implied Volatility (%)",
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font=dict(color="#b0b8c8"),
                xaxis=dict(gridcolor="#1e2228"),
                yaxis=dict(gridcolor="#1e2228"),
                height=320,
                margin=dict(l=10, r=10, t=10, b=40),
            )
            st.plotly_chart(fig_term, use_container_width=True)

            # Interpret shape
            first, last = term_df["atm_iv"].iloc[0], term_df["atm_iv"].iloc[-1]
            slope = last - first
            if abs(slope) < 1:
                shape_interp = "Flat curve — market expects similar volatility across all expiries."
            elif slope > 0:
                shape_interp = f"Upward-sloping (contango, +{slope:.1f}%) — normal. Market prices in more uncertainty further out."
            else:
                shape_interp = f"Inverted (backwardation, {slope:.1f}%) — stressed. Near-term vol exceeds long-term. Often signals a known event (earnings, macro)."

            st.markdown(
                f"<div class='info-box'>{shape_interp}</div>",
                unsafe_allow_html=True,
            )

            t1, t2, t3 = st.columns(3)
            with t1:
                st.metric("Front-month ATM IV", f"{first:.1f}%")
            with t2:
                st.metric("Back-month ATM IV", f"{last:.1f}%")
            with t3:
                st.metric("Slope", f"{slope:+.1f}%")
        else:
            st.info("Load more expiries to see term structure.")

    # ── IV Heatmap ──
    st.markdown("#### IV Heatmap — Strike × Expiry")
    st.markdown(
        "<div class='info-box'>Each cell shows the implied volatility for a given strike and expiry. "
        "Hover for exact values. This is the 2D projection of the 3D surface above.</div>",
        unsafe_allow_html=True,
    )

    if not iv_surface.empty:
        pivot = iv_surface.pivot_table(
            index="expiry", columns="strike", values="iv", aggfunc="mean"
        )
        pivot = pivot * 100  # convert to %
        pivot = pivot.sort_index()

        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[f"{k:.0f}" for k in pivot.columns],
            y=pivot.index.tolist(),
            colorscale=colorscale,
            hoverongaps=False,
            hovertemplate="Strike: %{x}<br>Expiry: %{y}<br>IV: %{z:.2f}%<extra></extra>",
            colorbar=dict(title="IV (%)"),
        ))
        fig_heat.add_vline(
            x=int(round(spot / 5.0) * 5),
            line_dash="dash", line_color="white",
            annotation_text=f"ATM ~{spot:.0f}",
        )
        fig_heat.update_layout(
            xaxis_title="Strike ($)",
            yaxis_title="Expiry Date",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="#b0b8c8"),
            height=350,
            margin=dict(l=10, r=10, t=10, b=40),
        )
        st.plotly_chart(fig_heat, use_container_width=True)


# ═════════════════════════════════════════════
# TAB 3 — GREEKS
# ═════════════════════════════════════════════
with tab_greeks:

    st.markdown(
        "<div class='info-box'>"
        "Greeks measure how sensitive an option's price is to different factors. "
        "Values shown below are for an <b>ATM call option</b> at the front-month expiry, "
        "computed from the Black-Scholes model."
        "</div>",
        unsafe_allow_html=True,
    )

    # Compute ATM greeks using BS closed-form
    atm_iv_val = atm_iv_front / 100 if atm_iv_front > 0 else 0.20
    T_front = atm_ivs[0]["days"] / 365 if atm_ivs else 30 / 365
    K_atm = spot

    def compute_greeks(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return {}
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega  = S * norm.pdf(d1) * np.sqrt(T) / 100   # per 1% IV move
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100   # per 1% rate
        return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

    greeks = compute_greeks(spot, K_atm, T_front, r, atm_iv_val)

    if greeks:
        g1, g2, g3, g4, g5 = st.columns(5)
        greek_meta = [
            ("Δ", "Delta",  greeks["delta"],  "#a78bfa", f"{greeks['delta']:.4f}"),
            ("Γ", "Gamma",  greeks["gamma"],  "#00c96b", f"{greeks['gamma']:.5f}"),
            ("Θ", "Theta",  greeks["theta"],  "#ff4d6d", f"${greeks['theta']:.4f}/day"),
            ("ν", "Vega",   greeks["vega"],   "#00d4aa", f"${greeks['vega']:.4f}/1%"),
            ("ρ", "Rho",    greeks["rho"],    "#ffb800", f"${greeks['rho']:.4f}/1%"),
        ]
        for col, (sym, name, val, color, label) in zip(
            [g1, g2, g3, g4, g5], greek_meta
        ):
            with col:
                st.markdown(
                    f"<div class='greek-card'>"
                    f"<div class='greek-symbol' style='color:{color}'>{sym}</div>"
                    f"<div class='greek-name'>{name}</div>"
                    f"<div class='greek-value' style='color:{color}'>{label}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<hr class='thin'>", unsafe_allow_html=True)

    # Greeks explained
    st.markdown("#### What each Greek tells you")
    cols_e = st.columns(2)
    explanations = [
        ("Δ Delta", f"{greeks.get('delta', 0):.4f}",
         "How much the option price moves for every $1 move in the stock.",
         f"If {ticker} rises $1, this ATM call gains ~${greeks.get('delta', 0):.2f} in value."),
        ("Γ Gamma", f"{greeks.get('gamma', 0):.5f}",
         "The rate of change of Delta. High near expiry or ATM. Shows how quickly your hedge needs rebalancing.",
         f"If {ticker} moves $1, Delta changes by {greeks.get('gamma', 0):.5f}."),
        ("Θ Theta", f"${greeks.get('theta', 0):.4f}/day",
         "Daily time decay. Option loses this much value each day purely from time passing.",
         f"This option loses ${abs(greeks.get('theta', 0)):.4f} per day with no stock movement."),
        ("ν Vega", f"${greeks.get('vega', 0):.4f}/1% IV",
         "Sensitivity to implied volatility. How much the option gains/loses per 1% IV move.",
         f"If IV rises 1% (e.g., {atm_iv_front:.0f}% → {atm_iv_front+1:.0f}%), this option gains ${greeks.get('vega', 0):.4f}."),
        ("ρ Rho", f"${greeks.get('rho', 0):.4f}/1% rate",
         "Sensitivity to interest rate changes. Less impactful for short-dated options.",
         f"If rates rise 1%, this call gains ${greeks.get('rho', 0):.4f}."),
    ]
    for i, (name, val, desc, example) in enumerate(explanations):
        with cols_e[i % 2]:
            st.markdown(
                f"<div class='edu-card' style='margin-bottom:12px'>"
                f"<h4>{name} = {val}</h4>"
                f"<p>{desc}</p>"
                f"<div class='edu-example'>📍 {example}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Greeks sensitivity chart
    st.markdown("#### Greeks Across Strikes (Front Month)")
    st.markdown(
        "<div class='info-box'>Shows how Delta and Gamma change as the stock moves "
        "through different strike prices. ATM options have the highest Gamma.</div>",
        unsafe_allow_html=True,
    )

    strike_range = np.linspace(spot * 0.80, spot * 1.20, 60)
    deltas, gammas = [], []
    for K in strike_range:
        g = compute_greeks(spot, K, T_front, r, atm_iv_val)
        deltas.append(g.get("delta", np.nan))
        gammas.append(g.get("gamma", np.nan))

    fig_greeks = make_subplots(specs=[[{"secondary_y": True}]])
    fig_greeks.add_trace(go.Scatter(
        x=strike_range, y=deltas,
        name="Delta (Δ)", line=dict(color="#a78bfa", width=2),
    ), secondary_y=False)
    fig_greeks.add_trace(go.Scatter(
        x=strike_range, y=gammas,
        name="Gamma (Γ)", line=dict(color="#00c96b", width=2, dash="dot"),
    ), secondary_y=True)
    fig_greeks.add_vline(x=spot, line_dash="dash", line_color="#ffffff44",
                         annotation_text="Spot", annotation_position="top")
    fig_greeks.update_layout(
        xaxis_title="Strike ($)",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#b0b8c8"),
        xaxis=dict(gridcolor="#1e2228"),
        yaxis=dict(gridcolor="#1e2228", title="Delta"),
        yaxis2=dict(gridcolor="#1e2228", title="Gamma", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
        height=340,
        margin=dict(l=10, r=10, t=10, b=40),
    )
    st.plotly_chart(fig_greeks, use_container_width=True)


# ═════════════════════════════════════════════
# TAB 4 — LEARN
# ═════════════════════════════════════════════
with tab_learn:

    st.markdown("## Understanding What These Numbers Mean")
    st.markdown(
        "<span style='color:#7a8090;font-size:13px'>"
        "New to options volatility? This section explains every concept shown in the dashboard.</span>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='thin'>", unsafe_allow_html=True)

    # ── Core concepts ──
    st.markdown("### Core Concepts")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
<div class="edu-card">
<h4>σ  Implied Volatility (IV)</h4>
<p>The market's collective forecast of how much a stock will move over a period — expressed as an
annualised percentage. It's called "implied" because it's not measured directly; it's <em>backed out</em>
of the market option price using the Black-Scholes model.</p>
<p><b>Higher IV = bigger expected moves.</b> Lower IV = calmer expectations.</p>
<div class="edu-example">
📍 IV of 20% → market implies a ±20% annual move.<br>
For a $100 stock that's roughly ±$1.26/day (±20% ÷ √252 trading days).
</div>
</div>
""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
<div class="edu-card">
<h4>∿  Volatility Skew</h4>
<p>In a perfect Black-Scholes world, puts and calls at equal distance from spot would have the same IV.
In reality, <b>puts trade at higher IV than calls</b> — this asymmetry is the "skew" or "smirk."</p>
<p>It reflects demand for downside protection. Positive skew (call > put) occasionally appears
when there's elevated squeeze risk or bullish positioning.</p>
<div class="edu-example">
📍 Skew of −3%: puts 10% OTM carry 3% more IV than calls 10% OTM.<br>
This is perfectly normal and expected in equity markets.
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("""
<div class="edu-card">
<h4>↗  Term Structure</h4>
<p>How IV changes with expiry. Normally IV is lower short-term and higher long-term (<b>contango</b>)
— more time means more uncertainty can accumulate.</p>
<p>During stress or before a known event (earnings, FOMC), the curve can <b>invert</b>: near-term vol
spikes above long-term vol.</p>
<div class="edu-example">
📍 Inverted curve: traders expect <em>more</em> volatility in the next 2 weeks than the next year.<br>
This usually signals a known binary event (earnings, CPI, Fed rate decision).
</div>
</div>
""", unsafe_allow_html=True)

    with c4:
        st.markdown("""
<div class="edu-card">
<h4>K/S  Moneyness</h4>
<p>Moneyness = Strike / Spot. It normalises the surface so you can compare across different
price levels and tickers.</p>
<ul style="color:#7a8090;font-size:13px;padding-left:16px;margin-top:8px">
<li>K/S &lt; 1.0 → out-of-the-money put</li>
<li>K/S = 1.0 → at-the-money (ATM)</li>
<li>K/S &gt; 1.0 → out-of-the-money call</li>
</ul>
<div class="edu-example">
📍 A $550 strike on a $587 stock has moneyness 0.938 — it's an OTM put.<br>
A $620 strike on the same stock has moneyness 1.056 — an OTM call.
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<hr class='thin'>", unsafe_allow_html=True)

    # ── Greeks ──
    st.markdown("### The Greeks — Option Sensitivities")
    st.markdown(
        "<div class='info-box'>Greeks measure how much an option's price moves in response to "
        "different market factors. Every options trader and market-maker monitors these continuously.</div>",
        unsafe_allow_html=True,
    )

    g_cols = st.columns(3)
    greek_cards = [
        ("Δ Delta", "#a78bfa",
         "How much the option price moves per $1 move in the stock. "
         "ATM call ≈ 0.50. Ranges 0–1 for calls, −1–0 for puts.",
         "Stock +$1 → ATM call gains ~$0.50. Stock −$1 → ATM put gains ~$0.50."),
        ("Γ Gamma", "#00c96b",
         "Rate of change of Delta. Peaks for ATM options near expiry. "
         "High Gamma = Delta changes fast as stock moves. Market makers must constantly re-hedge.",
         "Γ = 0.05: if stock rises $1, Delta goes from 0.50 → 0.55."),
        ("Θ Theta", "#ff4d6d",
         "Daily time decay — how much value the option loses each day purely from time passing. "
         "Option buyers pay Theta; option sellers collect it.",
         "Θ = −$0.08: option loses $0.08 per day with no stock movement."),
        ("ν Vega", "#00d4aa",
         "Sensitivity to implied volatility. Long options have positive Vega — rising IV is good for buyers. "
         "Expressed as change per 1% move in IV.",
         "ν = $0.25: if IV rises from 18% → 19%, option gains $0.25."),
        ("ρ Rho", "#ffb800",
         "Sensitivity to interest rate changes. Calls have positive Rho; puts negative. "
         "Usually the least impactful Greek, more relevant for long-dated options.",
         "ρ = $0.12: if rates rise 1%, call gains $0.12."),
        ("Vanna", "#fb923c",
         "Cross-sensitivity of Delta to IV changes (and Vega to stock price). "
         "Important for understanding skew dynamics and hedging near events.",
         "Positive Vanna: as IV rises, OTM calls gain more Delta than OTM puts lose."),
    ]
    for i, (name, color, desc, example) in enumerate(greek_cards):
        with g_cols[i % 3]:
            st.markdown(
                f"<div class='edu-card' style='margin-bottom:12px'>"
                f"<h4 style='color:{color}'>{name}</h4>"
                f"<p>{desc}</p>"
                f"<div class='edu-example'>📍 {example}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<hr class='thin'>", unsafe_allow_html=True)

    # ── Reading the surface ──
    st.markdown("### Reading Common Surface Patterns")

    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("""
<div class="pattern-card">
<div style="color:#ff4d6d;font-size:12px;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px">⚠ High Vol Regime</div>
<p style="font-size:13px;color:#b0c0d0">Surface is uniformly bright. IV above 30%+ across most strikes.
Market is pricing in large moves. Often seen during earnings season, macro events, or market dislocations.</p>
<div style="margin-top:12px;font-size:12px">
<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #1e2228">
<span style="color:#6a7080">1-week IV</span><span style="color:#ff4d6d;font-weight:600">55%</span></div>
<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #1e2228">
<span style="color:#6a7080">1-month IV</span><span style="color:#ffb800;font-weight:600">38%</span></div>
<div style="display:flex;justify-content:space-between;padding:4px 0">
<span style="color:#6a7080">6-month IV</span><span style="color:#ffb800;font-weight:600">32%</span></div>
</div>
</div>
""", unsafe_allow_html=True)

    with p2:
        st.markdown("""
<div class="pattern-card">
<div style="color:#00c96b;font-size:12px;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px">✓ Low Vol / Calm</div>
<p style="font-size:13px;color:#b0c0d0">Surface is mostly dim. IV below 15%. Market is complacent — often seen in slow,
trending bull markets. Low vol can be a contrarian warning: complacency can precede sharp moves.</p>
<div style="margin-top:12px;font-size:12px">
<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #1e2228">
<span style="color:#6a7080">1-week IV</span><span style="color:#00c96b;font-weight:600">10%</span></div>
<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #1e2228">
<span style="color:#6a7080">1-month IV</span><span style="color:#00c96b;font-weight:600">13%</span></div>
<div style="display:flex;justify-content:space-between;padding:4px 0">
<span style="color:#6a7080">6-month IV</span><span style="color:#00c96b;font-weight:600">16%</span></div>
</div>
</div>
""", unsafe_allow_html=True)

    with p3:
        st.markdown("""
<div class="pattern-card">
<div style="color:#ffb800;font-size:12px;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px">↑ Event Premium</div>
<p style="font-size:13px;color:#b0c0d0">Near-term IV spikes sharply for one specific expiry. Typically earnings,
FOMC meeting, or a macro data release. The spike crushes immediately after the event — called "vol crush."</p>
<div style="margin-top:12px;font-size:12px">
<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #1e2228">
<span style="color:#6a7080">1-week IV</span><span style="color:#ff4d6d;font-weight:600">85% ← spike</span></div>
<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #1e2228">
<span style="color:#6a7080">1-month IV</span><span style="color:#00c96b;font-weight:600">19%</span></div>
<div style="display:flex;justify-content:space-between;padding:4px 0">
<span style="color:#6a7080">6-month IV</span><span style="color:#00c96b;font-weight:600">21%</span></div>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<hr class='thin'>", unsafe_allow_html=True)

    # ── Methodology ──
    st.markdown("### How the Engine Works")
    with st.expander("Black-Scholes model & Brent IV solver"):
        st.markdown("""
**Step 1 — Fetch live option chain**
Real-time bid/ask prices are pulled from Yahoo Finance. Mid-price = (bid + ask) / 2.
Illiquid options (low bids) are filtered out before IV inversion.

**Step 2 — Black-Scholes pricing**
For a call option:

    d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
    d₂ = d₁ − σ√T
    C  = S·N(d₁) − K·e^(−rT)·N(d₂)

Where S = spot, K = strike, T = time to expiry, r = risk-free rate, σ = volatility.

**Step 3 — Implied volatility inversion**
Black-Scholes has no closed-form solution for σ given C. We solve numerically:

    BS(σ) − MarketPrice = 0

using **Brent's root-finding algorithm** (`scipy.optimize.brentq`), which combines bisection (stability)
with secant interpolation (speed). This is the industry-standard approach.

**Step 4 — Surface interpolation**
IVs across strikes and expiries are interpolated onto a smooth 3D grid using `scipy.interpolate.griddata`
with cubic interpolation (fallback to linear if cubic produces NaNs).
""")

    with st.expander("Limitations of this model"):
        st.markdown("""
- **Black-Scholes assumes constant volatility** — the surface itself shows this assumption is wrong
  (that's the whole point). The surface shows what vol *would need to be* to match market prices.
- **European-style pricing only** — American options can be exercised early; this model doesn't account for that.
- **Sparse far-OTM data** — interpolation artifacts appear where market data is thin.
- **Snapshot data, not streaming** — prices are cached for 5 minutes.
- **No stochastic vol calibration** — Heston or SABR models fit the surface more robustly.
""")

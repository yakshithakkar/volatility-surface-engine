import numpy as np
import plotly.graph_objects as go

from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# -----------------------------------
# Plot Volatility Surface
# -----------------------------------
def plot_vol_surface(iv_df, spot):

    # Safety copy
    iv_df = iv_df.copy()

    # -----------------------------------
    # Convert strike -> moneyness
    # -----------------------------------
    iv_df["moneyness"] = iv_df["strike"] / spot

    # Convert T -> days
    iv_df["days"] = iv_df["T"] * 365

    # -----------------------------------
    # Remove unrealistic IVs
    # -----------------------------------
    iv_df = iv_df[
        (iv_df["iv"] > 0.01) &
        (iv_df["iv"] < 1.5)
    ]

    # -----------------------------------
    # Create interpolation grid
    # -----------------------------------
    m_vals = np.linspace(
        iv_df["moneyness"].min(),
        iv_df["moneyness"].max(),
        60
    )

    t_vals = np.linspace(
        iv_df["days"].min(),
        iv_df["days"].max(),
        40
    )

    M, T_grid = np.meshgrid(m_vals, t_vals)

    # -----------------------------------
    # Interpolate IV surface
    # -----------------------------------
    Z = griddata(
        points=(
            iv_df["moneyness"],
            iv_df["days"]
        ),

        values=iv_df["iv"],

        xi=(M, T_grid),

        method="cubic"
    )

    # -----------------------------------
    # Fallback if cubic creates NaNs
    # -----------------------------------
    if np.isnan(Z).all():

        Z = griddata(
            points=(
                iv_df["moneyness"],
                iv_df["days"]
            ),

            values=iv_df["iv"],

            xi=(M, T_grid),

            method="linear"
        )

    # -----------------------------------
    # Build 3D surface
    # -----------------------------------
    fig = go.Figure(

        data=[

            go.Surface(
                x=M,
                y=T_grid,
                z=Z,

                colorscale="Viridis",

                colorbar=dict(
                    title="Implied Vol"
                )
            )

        ]

    )

    # -----------------------------------
    # Layout
    # -----------------------------------
    fig.update_layout(

        title="SPY Implied Volatility Surface",

        scene=dict(
            xaxis_title="Moneyness (K/S)",
            yaxis_title="Days to Expiry",
            zaxis_title="Implied Volatility"
        ),

        margin=dict(
            l=0,
            r=0,
            b=0,
            t=40
        ),

        height=750
    )

    return fig

    import matplotlib.pyplot as plt


# -----------------------------------
# Plot Volatility Smile
# -----------------------------------
def plot_smile(iv_df, target_expiry_label, spot):

    # Filter one expiry
    slice_df = iv_df[
        iv_df["expiry"] == target_expiry_label
    ].copy()

    # Convert strike -> moneyness
    slice_df["moneyness"] = (
        slice_df["strike"] / spot
    )

    # Sort properly
    slice_df = slice_df.sort_values(
        "moneyness"
    )

    # -----------------------------------
    # Plot
    # -----------------------------------
    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        slice_df["moneyness"],
        slice_df["iv"] * 100,

        "o-",

        linewidth=2,
        markersize=5
    )

    # ATM line
    ax.axvline(
        1.0,
        linestyle="--",
        alpha=0.6,
        label="ATM"
    )

    # Labels
    ax.set_xlabel(
        "Moneyness (K/S)"
    )

    ax.set_ylabel(
        "Implied Volatility (%)"
    )

    ax.set_title(
        f"Volatility Smile — {target_expiry_label}"
    )

    ax.legend()

    plt.tight_layout()

    # Save image
    plt.savefig(
        "vol_smile.png",
        dpi=150
    )

    plot.show()
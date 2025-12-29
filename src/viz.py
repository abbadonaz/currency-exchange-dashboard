from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Professional, muted palette for charts
COLOR_PALETTE = ["#0b3954", "#087E8B", "#5D6D7E", "#FFA630", "#E07A5F", "#9BB7D4", "#6A8D92"]

def render_kpis(metrics: dict[str, dict[str, float]], base: str) -> None:
    """Render compact KPI cards with colored 7-day delta arrows."""
    n = max(1, min(4, len(metrics)))
    cols = st.columns(n, gap="small")
    i = 0
    for cur, m in metrics.items():
        c = cols[i % n]
        latest = m.get("latest", float("nan"))
        chg7 = m.get("chg7", None)
        if chg7 is None or pd.isna(chg7):
            delta_str = "n/a"
            color = "#6c757d"
            arrow = ""
        else:
            arrow = "▲" if chg7 > 0 else ("▼" if chg7 < 0 else "")
            color = "#137333" if chg7 > 0 else ("#c92a2a" if chg7 < 0 else "#6c757d")
            delta_str = f"{arrow} {chg7:+.2f}%"

        card = f"""
        <div style="background:white;border-radius:10px;padding:12px;box-shadow:0 1px 4px rgba(11,57,84,0.06);">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <div style="font-weight:600;color:#0b3954">{cur}/{base}</div>
            <div style="font-size:12px;color:#9aa4ae">7d</div>
          </div>
          <div style="font-size:18px;font-weight:700;color:#0b3d4e;margin-bottom:6px;">{latest:.4f}</div>
          <div style="font-size:12px;color:{color};font-weight:600;">{delta_str}</div>
        </div>
        """
        c.markdown(card, unsafe_allow_html=True)
        i += 1

def plot_timeseries(rates: pd.DataFrame, base: str):
    df = rates.copy()
    df.index.name = "Date"
    df = df.reset_index().melt(id_vars="Date", var_name="Currency", value_name="Rate")
    fig = px.line(
        df,
        x="Date",
        y="Rate",
        color="Currency",
        title=f"Rates vs {base}",
        color_discrete_sequence=COLOR_PALETTE,
    )
    fig.update_layout(
        legend_title=None,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=40),
        font=dict(color="#0b3d4e"),
    )
    fig.update_yaxes(tickformat=".4f")
    return fig

def plot_returns_bar(returns: pd.DataFrame, base: str):
    df = returns.copy()
    df.index.name = "Date"
    df = df.reset_index().melt(id_vars="Date", var_name="Currency", value_name="Return")
    fig = px.bar(
        df,
        x="Date",
        y="Return",
        color="Currency",
        title=f"Daily % Change vs {base}",
        color_discrete_sequence=COLOR_PALETTE,
    )
    fig.update_layout(template="plotly_white", margin=dict(l=40, r=20, t=40, b=40), legend_title=None, hovermode="x unified", font=dict(color="#0b3d4e"))
    # Returns are fractional (e.g. 0.01 == 1%); show percent ticks
    fig.update_yaxes(tickformat=".2%")
    return fig

def plot_heatmap(returns: pd.DataFrame, base: str):
    corr = returns.corr().round(3)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdYlBu",
        zmin=-1,
        zmax=1,
        title=f"Correlation of Daily Returns (vs {base})",
    )
    fig.update_layout(template="plotly_white", margin=dict(l=40, r=20, t=40, b=40), font=dict(color="#0b3d4e"))
    return fig

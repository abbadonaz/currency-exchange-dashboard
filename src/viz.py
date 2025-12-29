from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def render_kpis(metrics: dict[str, dict[str, float]], base: str) -> None:
    n = max(1, min(4, len(metrics)))
    cols = st.columns(n)
    i = 0
    for cur, m in metrics.items():
        c = cols[i % n]
        change = "n/a" if pd.isna(m["chg7"]) else f"{m['chg7']:.2f}% (7d)"
        c.metric(f"{cur}/{base}", f"{m['latest']:.4f}", change)
        i += 1

def plot_timeseries(rates: pd.DataFrame, base: str):
    df = rates.copy()
    df.index.name = "Date"
    df = df.reset_index().melt(id_vars="Date", var_name="Currency", value_name="Rate")
    fig = px.line(df, x="Date", y="Rate", color="Currency", title=f"Rates vs {base}")
    fig.update_layout(legend_title=None, hovermode="x unified")
    return fig

def plot_returns_bar(returns: pd.DataFrame, base: str):
    df = returns.copy()
    df.index.name = "Date"
    df = df.reset_index().melt(id_vars="Date", var_name="Currency", value_name="Return")
    fig = px.bar(df, x="Date", y="Return", color="Currency", title=f"Daily % Change vs {base}")
    return fig

def plot_heatmap(returns: pd.DataFrame, base: str):
    corr = returns.corr().round(3)
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index))
    fig.update_layout(title=f"Correlation of Daily Returns (vs {base})")
    return fig

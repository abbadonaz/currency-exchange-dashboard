import streamlit as st
import pandas as pd
from src.config import settings
import plotly.graph_objects as go
from src import anomaly, data_sources, transform, viz

st.set_page_config(page_title="Currency Exchange Dashboard", layout="wide")

st.sidebar.title("Controls")
base_currency = st.sidebar.text_input("Base currency", settings.base_currency).upper()
targets = st.sidebar.multiselect(
    "Target currencies",
    ["USD","GBP","JPY","CHF","CNY","PLN","SEK","NOK","CAD","AUD"],
    default=["USD","GBP","JPY"],
)
n_days = st.sidebar.slider("Lookback (days)", min_value=30, max_value=365*3, value=365)
offline = st.sidebar.checkbox("Offline mode (use snapshot)", value=False)

st.title("💱 Currency Exchange Dashboard")
st.caption("Clean, testable, and extensible dashboard with analytics and forecasting hooks.")

@st.cache_data(show_spinner=False, ttl=settings.cache_ttl_min*60)
def load_rates(base: str, targets: list[str], days: int, offline_mode: bool) -> pd.DataFrame:
    df = data_sources.get_rates(base, days, offline=offline_mode)
    if targets:
        keep = [c for c in targets if c in df.columns]
        df = df[keep]
    return df

with st.spinner("Loading rates..."):
    rates = load_rates(base_currency, targets, n_days, offline)

if rates.empty:
    st.warning("No data returned. Check base currency or offline snapshot.")
    st.stop()

metrics = transform.compute_kpis(rates)
viz.render_kpis(metrics, base_currency)

tab_ts, tab_returns, tab_heat, tab_anom, tab_about = st.tabs(
    ["Time Series","% Change","Heatmap","Anomalies","About"]
)
with tab_anom:
    st.subheader("Anomaly Detection")

    col1, col2, col3 = st.columns(3)
    with col1:
        z_win = st.slider("Z-score window (days)", 10, 90, 30)
    with col2:
        z_thr = st.slider("Z-score threshold", 1.5, 4.0, 2.5, 0.1)
    with col3:
        contam = st.slider("IsolationForest contamination", 0.001, 0.10, 0.01, 0.001)

    # Compute flags
    z_flags = anomaly.rolling_zscore_anomalies(rates, window=z_win, z_thresh=z_thr)
    iso_flags = anomaly.isolation_forest_anomalies(rates, contamination=contam)

    st.markdown("**Latest anomaly snapshot (today):**")
    latest = pd.DataFrame({
        "Z-Score": z_flags.tail(1).T.iloc[:, 0],
        "IsolationForest": iso_flags.tail(1).T.iloc[:, 0],
    })
    st.dataframe(latest)

    cur = st.selectbox("Inspect currency", options=list(rates.columns), index=0, key="anom_cur")
    s = rates[cur].dropna()
    zf = z_flags[cur].reindex(s.index, fill_value=False)
    iff = iso_flags[cur].reindex(s.index, fill_value=False)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=f"{cur}/{base_currency}"))
    fig.add_trace(go.Scatter(
        x=s.index[zf], y=s.values[zf], mode="markers", name="Z-score",
        marker=dict(size=9, symbol="x")
    ))
    fig.add_trace(go.Scatter(
        x=s.index[iff], y=s.values[iff], mode="markers", name="IsolationForest",
        marker=dict(size=9, symbol="circle-open")
    ))
    fig.update_layout(title=f"Anomalies for {cur}/{base_currency}")
    st.plotly_chart(fig, use_container_width=True)

with tab_ts:
    st.plotly_chart(viz.plot_timeseries(rates, base_currency), use_container_width=True)

with tab_returns:
    returns = transform.pct_change(rates)
    st.plotly_chart(viz.plot_returns_bar(returns.tail(30), base_currency), use_container_width=True)

with tab_heat:
    returns = transform.pct_change(rates)
    st.plotly_chart(viz.plot_heatmap(returns, base_currency), use_container_width=True)

with tab_about:
    st.markdown("""
### About
- **Config** via environment variables (see `.env.example`) using pydantic-settings.
- **Caching**: Streamlit cache; optional offline snapshot fallback.
- **Structure**: `src/` modules for data, transforms, viz; tests under `tests/`.
- **Next**: add forecasting/backtesting in `src/analytics.py` and a Biotech Ops tab.
""")

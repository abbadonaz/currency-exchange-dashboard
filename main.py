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

    from src import news as newsmod, features as feat

with tab_anom:
    st.subheader("Anomaly Detection (Feature-aware)")

    # Controls
    colA, colB, colC, colD = st.columns(4)
    with colA:
        z_win = st.slider("Z-score window", 10, 90, 30)
    with colB:
        z_thr = st.slider("Z-score threshold", 1.5, 4.0, 2.5, 0.1)
    with colC:
        vol_win = st.slider("Vol window", 10, 120, 30)
    with colD:
        contam = st.slider("IF contamination", 0.001, 0.1, 0.01, 0.001)

    # Baseline: z-score on returns
    z_flags = anomaly.rolling_zscore_anomalies(rates, window=z_win, z_thresh=z_thr)

    # News sentiment fetch (RSS + Google News RSS)
    with st.expander("News sentiment (source feeds)", expanded=False):
        days_back = st.slider("News lookback (days)", 1, 30, 7, key="news_days")
        feeds_text = st.text_area(
            "RSS feeds (one per line",
            "\n".join(newsmod.DEFAULT_FEEDS), height=120
        )
        feed_list = [ln.strip() for ln in feeds_text.splitlines() if ln.strip()]
        if st.button("Fetch sentiment"):
            with st.spinner("Fetching & scoring news..."):
                df_news = newsmod.fetch_feeds(feed_list, days_back=days_back)
                st.write(f"Fetched {len(df_news)} items")
                st.dataframe(df_news[["published","title","sentiment","currencies","link"]], use_container_width=True)
                sent_daily = newsmod.aggregate_daily_sentiment(df_news)
                st.session_state["sent_daily"] = sent_daily
        sent_daily = st.session_state.get("sent_daily", pd.DataFrame())

    # Build per-currency feature tables: ret, vol, sent
    feats = feat.build_currency_features(rates, sent_daily, vol_window=vol_win)
    if_flags = anomaly.isolation_forest_per_currency(feats, contamination=contam)

    st.markdown("**Latest anomaly snapshot (today):**")
    latest = pd.DataFrame({
        "Z-Score": z_flags.tail(1).T.iloc[:, 0] if not z_flags.empty else [],
        "IF (ret+vol+sent)": if_flags.tail(1).T.iloc[:, 0] if not if_flags.empty else [],
    })
    st.dataframe(latest)

    cur_pick = st.selectbox("Inspect currency", list(rates.columns), index=0, key="anom_cur")
    s = rates[cur_pick].dropna()
    zf = z_flags[cur_pick].reindex(s.index, fill_value=False) if not z_flags.empty else s==False
    iff = if_flags[cur_pick].reindex(s.index, fill_value=False) if not if_flags.empty else s==False

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=f"{cur_pick}/{base_currency}"))
    fig.add_trace(go.Scatter(x=s.index[zf], y=s.values[zf], mode="markers", name="Z-score", marker=dict(size=9, symbol="x")))
    fig.add_trace(go.Scatter(x=s.index[iff], y=s.values[iff], mode="markers", name="IF (ret+vol+sent)", marker=dict(size=10, symbol="circle-open")))
    fig.update_layout(title=f"Anomalies for {cur_pick}/{base_currency}")
    st.plotly_chart(fig, use_container_width=True)

    st.caption("IsolationForest is trained **per currency** on features: return, rolling vol, and daily sentiment (0 if missing).")

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

## TODO: Add sentiment analysis to anomaly detection part , does it work with header or with entire text? is it reliable?
# TODO: make daily change vs eur respond to lookback 
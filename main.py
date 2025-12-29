# 💱 Currency Exchange Dashboard — License: MIT [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# See `LICENSE` for full terms.
import streamlit as st
import pandas as pd
from src.config import settings
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from src import anomaly, data_sources, transform, viz
from src.currency_calculator import convert_currency
st.set_page_config(page_title="Currency Exchange Dashboard", page_icon="💱", layout="wide", initial_sidebar_state="expanded")

st.sidebar.title("Controls")
base_currency = st.sidebar.text_input("Base currency", settings.base_currency).upper()
targets = st.sidebar.multiselect(
    "Target currencies",
    ["USD","GBP","JPY","CHF","CNY","PLN","SEK","NOK","CAD","AUD"],
    default=["PLN"],
)
n_days = st.sidebar.slider("Lookback (days)", min_value=30, max_value=365*3, value=365)
offline = st.sidebar.checkbox("Offline mode (use snapshot)", value=False)

st.markdown(
    """
    <div style="display:flex;align-items:center;gap:16px;margin-bottom:8px">
      <div style="font-size:44px">💱</div>
      <div>
        <h1 style="margin:0 0 4px 0">Currency Exchange Dashboard</h1>
        <p style="margin:0;color:#6c757d">Dashboard for currency exchange rates with anomaly detection.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f7fbfd 0%, #ffffff 100%); }
    .block-container { padding: 1rem 2rem; background: transparent; border-radius: 10px; }
    .stSidebar { background-color: #f8fafc; }
    h1 { color: #0b3d91; font-family: 'Inter', sans-serif; }
    p { color: #495057; font-family: 'Inter', sans-serif; }
    .stButton>button { background-color: #0b3d91; color: white; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

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
        # Let user scope feeds to a specific currency (or ALL)
        feed_options = ["ALL", base_currency] + list(rates.columns)
        # Ensure persistent selectbox with a stable key and reset if the previous value is now invalid
        if st.session_state.get("news_feed_currency") is not None and st.session_state.get("news_feed_currency") not in feed_options:
            st.session_state["news_feed_currency"] = "ALL"
        feed_currency = st.selectbox("Feed currency", options=feed_options, index=0, key="news_feed_currency")
        default_feeds = newsmod.default_feeds_for_currency(None if feed_currency == "ALL" else feed_currency, base_currency=base_currency)
        feeds_text = st.text_area(
            "RSS feeds (one per line)",
            "\n".join(default_feeds), height=120
        )
        feed_list = [ln.strip() for ln in feeds_text.splitlines() if ln.strip()]

        # Detect changed fetch params and clear cached sentiment when inputs change
        current_fetch_params = {
            "feed_currency": feed_currency,
            "feeds_text": feeds_text,
            "days_back": days_back,
            "base_currency": base_currency,
        }

        # Clear cached sentiment when the available rates (target currencies) change
        current_rates = list(rates.columns)
        last_rates = st.session_state.get("news_last_rates")
        if last_rates is not None and last_rates != current_rates:
            # Rates changed since last fetch; clear cached results and metadata
            if "sent_daily" in st.session_state:
                st.session_state.pop("sent_daily", None)
            st.session_state.pop("news_last_fetch_params", None)
            st.session_state.pop("news_last_fetch_ts", None)
            st.info("Available currencies changed — previous sentiment results were cleared. Click 'Fetch sentiment' to fetch updated results.")

        # If fetch params changed since the last fetch, clear cached results and metadata
        if st.session_state.get("news_last_fetch_params") is not None and st.session_state.get("news_last_fetch_params") != current_fetch_params:
            if "sent_daily" in st.session_state:
                st.session_state.pop("sent_daily", None)
            # clear stored metadata too (timestamp / params)
            st.session_state.pop("news_last_fetch_params", None)
            st.session_state.pop("news_last_fetch_ts", None)
            st.info("News feed parameters changed — previous sentiment results were cleared. Click 'Fetch sentiment' to fetch updated results.")

        if st.button("Fetch sentiment"):
            with st.spinner("Fetching & scoring news..."):
                filter_cur = None if feed_currency == "ALL" else feed_currency
                df_news = newsmod.fetch_feeds(feed_list, days_back=days_back, filter_currency=filter_cur)
                st.write(f"Fetched {len(df_news)} items (filter={filter_cur or 'ALL'})")
                st.dataframe(df_news[["published","title","sentiment","currencies","link"]], use_container_width=True)
                sent_daily = newsmod.aggregate_daily_sentiment(df_news)
                st.session_state["sent_daily"] = sent_daily
                st.session_state["news_last_fetch_params"] = current_fetch_params
                st.session_state["news_last_fetch_ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # record the rates snapshot used for this fetch so future changes can clear cache
                st.session_state["news_last_rates"] = current_rates
        sent_daily = st.session_state.get("sent_daily", pd.DataFrame())

        # Lightweight visual indicator of last fetched params & time
        if "sent_daily" in st.session_state and st.session_state.get("news_last_fetch_params"):
            last = st.session_state.get("news_last_fetch_ts")
            p = st.session_state.get("news_last_fetch_params", {})
            feeds_count = len(p.get("feeds_text", "").splitlines()) if p.get("feeds_text") else 0
            st.caption(f"Last fetched: {last} — Feed: {p.get('feed_currency','ALL')} • Days: {p.get('days_back')} • Feeds: {feeds_count}")

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

    # -- compute z-score series (for hover/annotations) aligned to s.index
    rets = s.pct_change()
    mu = rets.rolling(z_win).mean()
    sd = rets.rolling(z_win).std()
    z_series = ((rets - mu) / sd).reindex(s.index)

    zf = z_flags[cur_pick].reindex(s.index, fill_value=False) if not z_flags.empty else pd.Series(False, index=s.index)
    iff = if_flags[cur_pick].reindex(s.index, fill_value=False) if not if_flags.empty else pd.Series(False, index=s.index)

    from plotly.subplots import make_subplots

    # Build a 2-row subplot: rates (row 1) and z-score bars (row 2) for clarity
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.06,
        subplot_titles=(f"{cur_pick}/{base_currency}", "Z-score (returns)")
    )

    # Row 1: rates line
    fig.add_trace(
        go.Scatter(
            x=s.index,
            y=s.values,
            mode="lines",
            name=f"{cur_pick}/{base_currency}",
            line=dict(width=2, color="#0b3954"),
            hovertemplate="Date: %{x}<br>Rate: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Row 1: IF markers with detailed hover
    if iff.any():
        feats_df = feats.get(cur_pick, pd.DataFrame()).reindex(s.index).fillna(0.0)
        ret_vals = feats_df.get("ret", pd.Series(0.0, index=s.index)).values
        vol_vals = feats_df.get("vol", pd.Series(0.0, index=s.index)).values
        sent_vals = feats_df.get("sent", pd.Series(0.0, index=s.index)).values
        custom_if = np.vstack([ret_vals[iff], vol_vals[iff], sent_vals[iff]]).T
        fig.add_trace(
            go.Scatter(
                x=s.index[iff],
                y=s.values[iff],
                mode="markers",
                name="IF (ret+vol+sent)",
                marker=dict(size=10, symbol="circle-open", color="#FFA630"),
                customdata=custom_if,
                hovertemplate=(
                    "Date: %{x}<br>Rate: %{y:.4f}<br>IF anomaly: True"
                    "<br>Ret: %{customdata[0]:.2%}<br>Vol: %{customdata[1]:.2%}<br>Sent: %{customdata[2]:.2f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    # Row 2: z-score bars (color by sign)
    z_vals = z_series.fillna(0.0)
    colors = ["#E07A5F" if v > 0 else "#087E8B" for v in z_vals]
    fig.add_trace(
        go.Bar(
            x=s.index,
            y=z_vals,
            marker_color=colors,
            name="Z-score",
            hovertemplate="Date: %{x}<br>Z-score: %{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Add threshold lines on z-score subplot
    fig.add_hline(y=z_thr, line=dict(color="rgba(224,122,95,0.6)", dash="dash"), row=2, col=1)
    fig.add_hline(y=-z_thr, line=dict(color="rgba(8,126,139,0.6)", dash="dash"), row=2, col=1)

    # Annotate top N absolute z anomalies on z subplot for clarity
    top_n = 3
    top_z = z_vals[ zf ].abs().nlargest(top_n)
    for idx, val in top_z.items():
        z_val = z_series.loc[idx]
        sign_color = "#E07A5F" if z_val > 0 else "#087E8B"
        bg = "rgba(224,122,95,0.12)" if z_val > 0 else "rgba(8,126,139,0.12)"
        fig.add_annotation(
            x=idx,
            y=z_val,
            xref='x',
            yref='y2',
            text=f"Z={z_val:.2f}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-20,
            bgcolor=bg,
            bordercolor=sign_color,
            font=dict(color=sign_color),
        )

    fig.update_layout(
        title=f"Anomalies for {cur_pick}/{base_currency}",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40)
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, row=1, col=1)
    fig.update_yaxes(title_text="Z-score", row=2, col=1)

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
st.subheader("💱 Currency Converter")

latest_rates = rates.dropna().iloc[-1].to_dict()
latest_rates[base_currency] = 1.0

def swap_currencies():
    st.session_state.from_currency, st.session_state.to_currency = (
        st.session_state.to_currency,
        st.session_state.from_currency,
    )

if "from_currency" not in st.session_state:
    st.session_state.from_currency = base_currency

if "to_currency" not in st.session_state:
    st.session_state.to_currency = sorted(latest_rates.keys())[0]

col1, col2, col3 = st.columns(3)

with col1:
    amount = st.number_input(
        "Amount",
        min_value=0.0,
        value=100.0,
        step=10.0,
        key="conv_amount"
    )

with col2:
    st.selectbox(
        "From",
        options=sorted(latest_rates.keys()),
        key="from_currency"
    )

with col3:
    st.selectbox(
        "To",
        options=sorted(latest_rates.keys()),
        key="to_currency"
    )

st.button("🔄 Swap currencies", on_click=swap_currencies)

from_currency = st.session_state.from_currency
to_currency = st.session_state.to_currency

converted = convert_currency(
    amount=amount,
    from_currency=from_currency,
    to_currency=to_currency,
    exchange_rates=latest_rates
)

st.metric(
    label=f"{from_currency} → {to_currency}",
    value=f"{converted:,.2f} {to_currency}"
)

fee_pct = st.slider("Bank fee (%)", 0.0, 5.0, 0.5)

after_fee = converted * (1 - fee_pct / 100)

st.metric(
    label="After Fees",
    value=f"{after_fee:,.2f} {to_currency}"
)

st.caption(f"Rates as of {rates.index[-1].date()} (base: {base_currency})")


# TODO: make daily change vs eur respond to lookback 
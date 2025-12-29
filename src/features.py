from __future__ import annotations
import pandas as pd

def rolling_volatility(returns: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Rolling std of daily returns per currency."""
    return returns.rolling(window).std()

def build_currency_features(
    rates: pd.DataFrame,
    sentiment_daily: pd.DataFrame | None,
    vol_window: int = 30,
) -> dict[str, pd.DataFrame]:
    """
    For each currency, build a per-day feature table with columns:
    ['ret', 'vol', 'sent'] aligned on dates.
    Returns a dict: currency -> DataFrame(features)
    """
    rets = rates.pct_change()
    vol = rolling_volatility(rets, vol_window)

    # Sentiment: (date, currency) -> mean_sentiment
    sent_map: dict[tuple, float] = {}
    if sentiment_daily is not None and not sentiment_daily.empty:
        for _, r in sentiment_daily.iterrows():
            sent_map[(r["date"], r["currency"])] = float(r["mean_sentiment"])

    features = {}
    for cur in rates.columns:
        df = pd.DataFrame(index=rates.index)
        df["ret"] = rets[cur]
        df["vol"] = vol[cur]
        # align sentiment by calendar date (index may be tz-naive)
        sents = []
        for d in df.index:
            key = (d.date(), cur)
            sents.append(sent_map.get(key, 0.0))  # missing -> neutral 0.0
        df["sent"] = sents
        features[cur] = df.dropna(subset=["ret","vol"])  # keep rows with core features
    return features

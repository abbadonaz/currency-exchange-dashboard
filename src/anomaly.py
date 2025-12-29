from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

def isolation_forest_anomalies(rates: pd.DataFrame, contamination: float = 0.01) -> pd.DataFrame:
    """Boolean DF using IsolationForest on daily returns."""
    rets = rates.pct_change().dropna()
    if rets.empty:
        return pd.DataFrame(False, index=rates.index, columns=rates.columns)
    model = IsolationForest(
        contamination=contamination, max_samples=0.66, random_state=42, n_estimators=200, n_jobs=-1
    )
    preds = model.fit_predict(rets.values)  # -1 anomalous, 1 normal
    row_flags = pd.Series(preds == -1, index=rets.index)
    flags = pd.DataFrame(False, index=rets.index, columns=rets.columns)
    flags.loc[row_flags[row_flags].index, :] = True
    # Reindex to full rate index
    return flags.reindex(rates.index, fill_value=False)

def rolling_zscore_anomalies(rates: pd.DataFrame, window: int = 30, z_thresh: float = 2.5) -> pd.DataFrame:
    """Boolean DF: True where |z-score| of daily returns >= threshold."""
    rets = rates.pct_change()
    mu = rets.rolling(window).mean()
    sd = rets.rolling(window).std()
    z = (rets - mu) / sd
    return (z.abs() >= z_thresh).fillna(False)

def isolation_forest_per_currency(
    features: dict[str, pd.DataFrame],
    contamination: float = 0.01,
) -> pd.DataFrame:
    """
    Train one IsolationForest per currency on columns ['ret','vol','sent'].
    Returns a boolean DataFrame flags indexed by date, columns=currency.
    """
    # Union all dates to a common index
    all_idx = None
    for df in features.values():
        all_idx = df.index if all_idx is None else all_idx.union(df.index)
    flags = pd.DataFrame(False, index=all_idx, columns=list(features.keys()))
    for cur, df in features.items():
        if df.shape[0] < 30:
            continue
        X = df[["ret","vol","sent"]].values
        model = IsolationForest(
            contamination=contamination, n_estimators=300, random_state=42, n_jobs=-1
        )
        preds = model.fit_predict(X)  # -1 anomalous
        f = pd.Series(preds == -1, index=df.index)
        flags.loc[df.index, cur] = f
    return flags.sort_index()
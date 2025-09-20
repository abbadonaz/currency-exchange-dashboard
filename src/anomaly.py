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
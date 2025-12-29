from __future__ import annotations
import pandas as pd
import numpy as np

def pct_change(rates: pd.DataFrame) -> pd.DataFrame:
    """Return daily percentage changes."""
    return rates.pct_change().dropna(how="all")

def compute_kpis(rates: pd.DataFrame) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for col in rates.columns:
        s = rates[col].dropna()
        if len(s) < 10:
            continue
        latest = float(s.iloc[-1])
        chg7 = float((s.iloc[-1] / s.iloc[-8] - 1) * 100) if len(s) > 7 else float("nan")
        vol90 = (
            float(s.pct_change().rolling(90).std().iloc[-1] * (252**0.5) * 100)
            if len(s) > 100 else float("nan")
        )
        metrics[col] = {"latest": latest, "chg7": chg7, "vol90": vol90}
    return metrics


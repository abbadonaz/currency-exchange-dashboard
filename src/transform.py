from __future__ import annotations
import pandas as pd
import numpy as np

def pct_changes(rates: pd.DataFrame) -> pd.DataFrame:
    """Compute daily pepcentage of changes."""
    return rates.pct_change().dropna(how="all")
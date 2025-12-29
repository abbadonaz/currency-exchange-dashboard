from __future__ import annotations
import io
import pandas as pd
import requests
from .config import settings

ECB_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"

def _load_ecb_hist() -> pd.DataFrame:
    """Load historical ECB reference data (daily, base EUR)."""
    resp = requests.get(ECB_URL, timeout=30)
    resp.raise_for_status()
    z = resp.content
    import zipfile
    with zipfile.ZipFile(io.BytesIO(z)) as zf:
        with zf.open("eurofxref-hist.csv") as f:
            df = pd.read_csv(f)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")
        df["EUR"] = 1.0
        return df
    
def _to_base(df_eur_base: pd.DataFrame, base: str) -> pd.DataFrame:
    """Convert from EUR base to an arbitrary base currency."""
    if base not in df_eur_base.columns:
        raise ValueError(f"Base currency {base} not in ECB table.")
    base_series = df_eur_base[base]
    return df_eur_base.divide(base_series, axis=0)

def get_rates(base_currency: str, days: int = 365, offline: bool = False) -> pd.DataFrame:
    """Return last N days of daily rates in the given base currency."""
    if offline:
        try:
            df = pd.read_parquet(settings.snapshot_path)
            df.index = pd.to_datetime(df.index)
        except Exception:
            return pd.DataFrame()
    else:
        df = _load_ecb_hist()
    df = df.ffill()
    df_base = _to_base(df, base_currency)
    if days is not None:
        cutoff = df_base.index.max() - pd.Timedelta(days=days)
        df_base = df_base[df_base.index >= cutoff]
    return df_base
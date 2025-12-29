from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
import feedparser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



def default_feeds_for_currency(currency: str | None = None, base_currency: str | None = None) -> list[str]:
    """Return a list of recommended RSS feeds for the given currency.

    If `currency` is None, returns the module DEFAULT_FEEDS. If provided, this
    will add Google News RSS searches scoped to the currency and (optionally)
    the base currency (for pair searches).
    """
    feeds = [
        "https://www.ecb.europa.eu/press/pressconf/pressconf.rss",
        "https://www.ecb.europa.eu/press/pressreleases/rdf/pressreleases.rss",
        f"https://news.google.com/rss/search?q={currency}+exchange+rate",
        f"https://news.google.com/rss/search?q={currency}+currency",
    ]
    if base_currency:
        # add pair-specific search (e.g., EUR+USD+exchange+rate)
        pair = f"{base_currency}+{currency}+exchange+rate"
        feeds.append(f"https://news.google.com/rss/search?q={pair}")
    # Deduplicate while preserving order
    seen = set()
    out = []
    for f in feeds:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out

CURRENCY_KEYS = {
    "USD": ["usd", "dollar", "us dollar", "u.s. dollar"],
    "EUR": ["eur", "euro"],
    "GBP": ["gbp", "pound", "sterling"],
    "JPY": ["jpy", "yen"],
    "CHF": ["chf", "swiss franc"],
    "CNY": ["cny", "yuan", "renminbi"],
    "CAD": ["cad", "loonie"],
    "AUD": ["aud", "aussie"],
    "PLN": ["pln", "zloty", "złoty"],
    "SEK": ["sek", "krona"],
    "NOK": ["nok", "krone"],
}

analyzer = SentimentIntensityAnalyzer()

@dataclass
class NewsItem:
    published: pd.Timestamp
    title: str
    summary: str
    link: str
    sentiment: float
    currencies: list[str]

def _infer_currencies(text:str) -> list[str]:
    t = text.lower()
    hits = []
    for cur, keys in CURRENCY_KEYS.items():
        if any(k in t for k in keys):
            hits.append(cur)
    return sorted(set(hits))
def fetch_feeds(
    feeds: list[str] | None = None,
    days_back: int = 7,
    filter_currency: str | None = None,
) -> pd.DataFrame:
    """Fetch RSS feeds and return scored items.

    If `filter_currency` is provided, only items that mention that currency
    (based on `_infer_currencies`) will be returned.
    """
    if feeds is None:
        feeds = DEFAULT_FEEDS
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days_back)
    items: list[NewsItem] = []
    for url in feeds:
        parsed = feedparser.parse(url)
        for e in parsed.entries:
            pub = None
            for key in ("published_parsed", "updated_parsed"):
                if getattr(e, key, None):
                    pub = pd.Timestamp(dt.datetime(*getattr(e, key)[:6]), tz="UTC")
                    break
            if pub is None:
                pub = pd.Timestamp.utcnow()
            if pub < cutoff:
                continue
            title = getattr(e, "title", "") or ""
            summary = getattr(e, "summary", "") or ""
            text = f"{title}. {summary}"
            sent = analyzer.polarity_scores(text)["compound"]  # the overall sentiment score between -1 and 1
            link = getattr(e, "link", "") or ""
            curs = _infer_currencies(text)
            # If a filter is requested, skip items that don't mention the currency.
            if filter_currency and filter_currency not in curs:
                continue
            items.append(NewsItem(pub, title, summary, link, sent, curs))
    if not items:
        return pd.DataFrame(columns=["published","title","summary","link","sentiment","currencies"])
    df = pd.DataFrame([i.__dict__ for i in items]).sort_values("published", ascending=False)
    return df

def aggregate_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand rows by currency and compute mean sentiment per currency per day.
    Output columns: date, currency, mean_sentiment, n
    """
    if df.empty:
        return pd.DataFrame(columns=["date","currency","mean_sentiment","n"])
    rows = []
    for _, r in df.iterrows():
        for c in (r["currencies"] or []):
            rows.append({"date": r["published"].date(), "currency": c, "sentiment": r["sentiment"]})
    if not rows:
        return pd.DataFrame(columns=["date","currency","mean_sentiment","n"])
    dd = pd.DataFrame(rows)
    out = dd.groupby(["date","currency"])["sentiment"].agg(["mean","count"]).reset_index()
    out = out.rename(columns={"mean": "mean_sentiment", "count": "n"})
    return out
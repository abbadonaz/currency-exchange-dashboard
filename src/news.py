from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
import feedparser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DEFAULT_FEEDS = [
    "https://www.ecb.europa.eu/press/pressconf/pressconf.rss",
    "https://www.ecb.europa.eu/press/pressreleases/rdf/pressreleases.rss",
    "https://news.google.com/rss/search?q=EUR+USD+exchange+rate",
    "https://news.google.com/rss/search?q=currency+exchange+ECB",
]

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
def fetch_feeds(feeds: list[str] | None = None, days_back: int = 7) -> pd.DataFrame:
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
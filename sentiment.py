import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize sentiment analyzer
_analyzer = SentimentIntensityAnalyzer()

def score_text(text: str) -> float:
    """
    Compute compound sentiment score for a given text.
    Range: -1 (very negative) to +1 (very positive)
    """
    if not text:
        return 0.0
    return float(_analyzer.polarity_scores(text)["compound"])



def compute_daily_sentiment_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes average sentiment per day.
    Expects a DataFrame with at least 'published_at' and 'sentiment' columns.
    """

    if "published_at" not in df.columns or "sentiment" not in df.columns:
        return pd.DataFrame(columns=["date", "sentiment"])

    # Parse timestamps
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")

    # Clean sentiment values
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")

    # Drop missing or invalid rows
    df = df.dropna(subset=["published_at", "sentiment"])

    # Group by date
    df["date"] = df["published_at"].dt.date
    result = df.groupby("date", as_index=False)["sentiment"].mean()
    result.columns = ["date", "sentiment"]

    return result


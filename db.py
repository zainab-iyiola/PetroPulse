from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import pandas as pd
import os

# Path to SQLite database file
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed", "insights.sqlite")

def get_engine() -> Engine:
    """Create a database engine."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return create_engine(f"sqlite:///{DB_PATH}", future=True)

def init_db(engine: Engine):
    """Initialize the database with the articles table."""
    with engine.begin() as conn:
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                title TEXT,
                url TEXT UNIQUE,
                authors TEXT,
                published_at TIMESTAMP,
                text TEXT,
                sentiment REAL,
                entities TEXT,
                topics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        '''))

def insert_articles(engine: Engine, rows: list[dict]):
    if not rows:
        return 0
    df = pd.DataFrame(rows).dropna(subset=["url"]).drop_duplicates(subset=["url"])

    # Remove any URLs that already exist in the DB
    with engine.begin() as conn:
        try:
            existing = pd.read_sql("SELECT url FROM articles", con=conn)["url"].tolist()
        except Exception:
            existing = []
    df = df[~df["url"].isin(existing)]

    if df.empty:
        return 0

    df.to_sql("articles", con=engine, if_exists="append", index=False)
    return len(df)


def get_articles_df(engine: Engine) -> pd.DataFrame:
    """Retrieve all articles from the database."""
    try:
        df = pd.read_sql("SELECT * FROM articles", con=engine)
        if "published_at" in df.columns:
            df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


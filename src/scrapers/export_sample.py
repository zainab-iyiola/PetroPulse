import pandas as pd
from src.storage.db import get_engine

# Connect to your SQLite database
engine = get_engine()

# Read all articles into a DataFrame
df = pd.read_sql("articles", engine)

# Export to CSV
df.to_csv("data/sample_articles.csv", index=False)

print(f"✅ Exported {len(df)} articles to data/sample_articles.csv")

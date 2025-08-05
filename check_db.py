# scripts/check_db.py
import sqlite3
from tabulate import tabulate

DB_PATH = r"data/processed/insights.sqlite"

with sqlite3.connect(DB_PATH) as con:
    # List tables
    tables = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    print("Tables:", tables)

    # Preview articles table
    if ('articles',) in tables:
        rows = con.execute("SELECT source, title, url, published_at FROM articles LIMIT 5").fetchall()
        print("\nSample rows from 'articles':")
        print(tabulate(rows, headers=["Source", "Title", "URL", "Published At"], tablefmt="pretty"))
    else:
        print("No 'articles' table found.")

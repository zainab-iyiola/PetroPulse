import sqlite3

DB_PATH = "data/processed/insights.sqlite"

with sqlite3.connect(DB_PATH) as conn:
    cursor = conn.cursor()
    # Check if 'content' column exists
    cursor.execute("PRAGMA table_info(articles)")
    columns = [col[1] for col in cursor.fetchall()]
    if "content" not in columns:
        print("Adding 'content' column...")
        cursor.execute("ALTER TABLE articles ADD COLUMN content TEXT")
        conn.commit()
        print("Column added.")
    else:
        print("'content' column already exists.")

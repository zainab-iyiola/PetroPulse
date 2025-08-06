#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from datetime import datetime, timedelta, timezone
from typing import Dict, List
import pandas as pd
from tqdm import tqdm
from textblob import TextBlob
from newspaper import Article

from src.storage.db import get_engine, insert_articles
from src.scrapers.news_scraper import fetch_rss_entries
from src.nlp.entities import extract_org_gpe, entities_to_json

# Unified topic list (for both ingestion and dashboard filtering)
TOPIC_LIST = [
    "Exploration", "Seismic Surveys", "Reservoir Engineering", "Drilling", "Well Logging",
    "Well Intervention", "Well Completion", "Hydraulic Fracturing", "Production Optimization",
    "Enhanced Oil Recovery", "Shale Gas", "Oil Sands", "Deepwater", "Offshore Drilling",
    "FPSO", "Directional Drilling", "Mudlogging", "Formation Damage", "Core Flooding",
    "Pipeline Transportation", "Pipeline Safety", "Gas Processing", "Liquefied Natural Gas",
    "Storage", "Compressor Stations", "Metering and SCADA", "Crude Transport",
    "Permian Basin", "Hydrogen Blending", "CNG", "Sour Gas", "Sweet Gas",
    "Refining", "Petrochemicals", "Retail Fuels", "LNG Export", "Crude Oil Pricing",
    "Trading and Supply", "Turnarounds", "Sulfur Recovery",
    "Carbon Capture", "CCUS", "Hydrogen", "Blue Hydrogen", "Green Hydrogen",
    "Methane Emissions", "Flaring Reduction", "Energy Transition", "Decarbonization",
    "Net Zero", "Carbon Markets", "CO2 Sequestration", "CO2 Storage", "Greenhouse Gas",
    "Wind Energy", "Solar Integration", "Geothermal", "Biofuels", "Hybrid Energy Systems",
    "Photovoltaics", "Tidal Energy", "Clean Energy", "Renewable Integration",
    "Digital Oilfield", "AI in Energy", "Machine Learning", "Predictive Maintenance",
    "Remote Monitoring", "Blockchain in Oil and Gas", "Automation", "IoT in Energy",
    "Subsurface Modeling", "Big Data", "Edge Computing", "Cloud", "Digital Twin",
    "Data Science", "Deep Learning", "Image Analysis", "Robotics",
    "Porosity", "Permeability", "Petrophysics", "Fluid Saturation", "Capillary Pressure",
    "Asphaltene", "Wax Deposition",
    "Electricity", "Smart Grid", "Load Forecasting", "Power Generation",
    "Grid Resilience", "Distributed Energy", "Energy Efficiency",
    "Oil Prices", "Energy Policy", "Energy Markets", "Energy Security",
    "Environmental Compliance", "Inflation Reduction Act", "Energy Investment",
    "Carbon Disclosure", "Emissions Reporting", "OPEC", "IEA", "IRA",
    "Energy Innovation", "Workforce Transition", "Environmental Social Governance",
    "ESG", "Safety", "HSE", "Strategic Reserve"
]

ENERGY_KEYWORDS = [t.lower() for t in TOPIC_LIST]

def extract_full_content(url: str) -> str:
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text.strip()
    except Exception as e:
        print(f"Failed to extract content from {url}: {e}")
        return ""

def is_energy_related(text: str) -> bool:
    text = text.lower()
    return any(keyword in text for keyword in ENERGY_KEYWORDS)

def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ingest latest energy news into the database.")
    p.add_argument("--days", type=int, default=30, help="Only keep items from last N days")
    p.add_argument("--per-feed", type=int, default=30, help="Max items per feed")
    return p

def filter_by_date(items: List[Dict], days: int) -> List[Dict]:
    cutoff = _as_utc(datetime.now(timezone.utc) - timedelta(days=days))
    out = []
    for it in items:
        dt = it.get("published_at")
        if isinstance(dt, datetime):
            if _as_utc(dt) >= cutoff:
                out.append(it)
        else:
            out.append(it)
    return out

def main(days: int, per_feed: int) -> None:
    print(f"[ingest] Fetching RSS items (per feed: {per_feed}) …")
    items = fetch_rss_entries(limit_per_feed=per_feed)
    if not items:
        print("[ingest] No items found.")
        return

    items = filter_by_date(items, days=days)
    df = pd.DataFrame(items)
    if "url" not in df.columns:
        print("[ingest] No 'url' field found.")
        return

    before = len(df)
    df = df.dropna(subset=["url"]).drop_duplicates(subset=["url"])
    after = len(df)
    print(f"[ingest] Items after dedupe: {after} (removed {before - after})")

    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")

    rows: List[Dict] = []
    print("[ingest] Fetching full article content & extracting entities …")
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Fetching articles"):
        url = r.get("url", "")
        title = r.get("title", "")
        source = r.get("source", "")
        summary = r.get("summary", "")
        published_at = r.get("published_at")

        text = extract_full_content(url)

        combined_text = f"{title} {summary} {text}"
        if not is_energy_related(combined_text):
            continue  # Skip non-energy articles

        orgs, gpes = extract_org_gpe(text, title=title)
        ents_json = entities_to_json(orgs, gpes)

        sentiment_score = None
        if text:
            try:
                sentiment_score = TextBlob(text).sentiment.polarity
            except Exception:
                sentiment_score = None

        rows.append({
            "source": source,
            "title": title,
            "url": url,
            "published_at": published_at,
            "content": text,
            "entities": ents_json,
            "sentiment": sentiment_score,
            "topics": None,
        })

    if not rows:
        print("[ingest] Nothing to insert (no energy-related articles).")
        return

    engine = get_engine()
    print(f"[ingest] Inserting {len(rows)} articles into the database …")
    df_out = pd.DataFrame(rows)
    try:
        insert_articles(engine, df_out.to_dict(orient="records"))
        print(f"[ingest] Inserted {len(df_out)} articles.")
    except Exception as e:
        print(f"[ingest] Error during insert: {e}")

if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(days=args.days, per_feed=args.per_feed)

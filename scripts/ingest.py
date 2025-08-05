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
from textblob import TextBlob  # for sentiment

from src.storage.db import get_engine, insert_articles
from src.scrapers.news_scraper import fetch_rss_entries, fetch_article_text
from src.nlp.entities import extract_org_gpe, entities_to_json


# Define keywords related to oil, gas, and energy
ENERGY_KEYWORDS = [
    # ⚙️ Upstream, Midstream, Downstream
    "oil", "gas", "petroleum", "refinery", "exploration", "production", "drilling",
    "completion", "fracturing", "reservoir", "perforation", "wellbore", "wellhead",
    "upstream", "midstream", "downstream", "flaring", "seismic", "mudlogging",
    
    # 🛢️ Equipment & Operations
    "pipeline", "compressor", "separator", "flow assurance", "subsea", "FPSO",
    "offshore", "onshore", "platform", "rig", "casing", "coiled tubing",
    
    # 💨 Natural Gas, LNG, Hydrogen, and Storage
    "natural gas", "methane", "lng", "cng", "gas processing", "hydrogen", "h2",
    "blue hydrogen", "green hydrogen", "underground hydrogen storage", "uhs", "salt cavern",
    "gas storage", "strategic petroleum reserve", "sour gas", "sweet gas",
    
    # 🌿 Renewables and Energy Transition
    "renewables", "solar", "wind", "photovoltaic", "geothermal", "tidal", "biomass",
    "clean energy", "green energy", "energy transition", "decarbonization", "sustainability",
    "net zero", "energy mix", "renewable integration",

    # 🌍 Carbon Management & Climate
    "carbon", "ccs", "ccus", "carbon capture", "carbon dioxide", "co2 storage", 
    "co2 sequestration", "carbon footprint", "carbon trading", "carbon intensity",
    "methane emissions", "greenhouse gas", "ghg emissions", "climate change",

    # 🔌 Power Systems & Grid
    "energy", "power", "electricity", "smart grid", "energy efficiency",
    "load forecasting", "grid resilience", "power generation", "distributed energy",

    # 📊 Digital & Emerging Technologies
    "ai", "artificial intelligence", "ml", "machine learning", "data science",
    "digital twin", "iot", "internet of things", "cloud", "edge computing",
    "predictive maintenance", "data-driven", "big data", "analytics", "deep learning",
    "robotics", "automation", "image analysis", "remote sensing",

    # 🧪 Subsurface & Science
    "core flooding", "porosity", "permeability", "petrophysics", "fluid saturation",
    "capillary pressure", "formation damage", "asphaltene", "wax deposition",

    # 💸 Policy, Investment & Economy
    "oil price", "brent", "wti", "energy policy", "opec", "iea", "epc", "supply chain",
    "energy markets", "strategic reserve", "subsidy", "inflation reduction act", 
    "ira", "energy bill", "fossil fuel", "energy investment",

    # 📄 ESG & Industry Trends
    "esg", "environmental social governance", "safety", "hse", "emissions reporting",
    "carbon disclosure", "environmental compliance", "energy innovation", 
    "workforce transition", "digital transformation", "energy workforce"
]


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
    print("[ingest] Fetching article text & extracting entities …")
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Fetching articles"):
        url = r.get("url", "")
        title = r.get("title", "")
        source = r.get("source", "")
        summary = r.get("summary", "")
        published_at = r.get("published_at")

        try:
            text = fetch_article_text(url) or ""
        except Exception:
            text = ""

        # Combine title, summary, and body for filtering
        content = f"{title} {summary} {text}"
        if not is_energy_related(content):
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
            "text": text,
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

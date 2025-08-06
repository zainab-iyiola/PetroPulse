#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

# 1) Make sure the repo root (one level above "app/") is on PYTHONPATH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Now we can safely import from src.*
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from datetime import datetime
from zoneinfo import ZoneInfo
from io import StringIO
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 2) Guard our project‐specific imports
try:
    from src.storage.db import get_engine, get_articles_df
except ModuleNotFoundError:
    st.error("Could not import your database module; make sure src/storage/db.py is present.")
    st.stop()

try:
    from src.models.sentiment import compute_daily_sentiment_index
except ModuleNotFoundError:
    compute_daily_sentiment_index = None

try:
    from src.nlp.pipeline import extract_entities
except ModuleNotFoundError:
    extract_entities = lambda x: []

# Timezone helper
EASTERN = ZoneInfo("US/Eastern")
def to_edtz(ts):
    ts = pd.to_datetime(ts, utc=True, errors="coerce")
    if ts is pd.NaT:
        return ""
    return ts.tz_convert(EASTERN).strftime("%Y-%m-%d %H:%M")

# Small KPI card
def kpi_card(label: str, value: str, gradient: str):
    st.markdown(
        f"""
        <div style="
            background: {gradient};
            padding: 16px 20px;
            border-radius: 12px;
            color: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        ">
          <div style="opacity:0.7; font-size:14px;">{label}</div>
          <div style="font-size:26px; font-weight:600; margin-top:4px;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Sentiment trend chart
def sentiment_chart(df_idx: pd.DataFrame, title: str):
    if df_idx.empty:
        st.info("No sentiment data in this range.")
        return
    chart = (
        alt.Chart(df_idx)
        .mark_line(point=True)
        .encode(
            x="date:T",
            y=alt.Y("sentiment:Q", scale=alt.Scale(domain=(-1,1))),
            tooltip=["date:T","sentiment:Q"],
        )
        .properties(height=200)
        .interactive()
    )
    st.subheader(title)
    st.altair_chart(chart, use_container_width=True)

# ─── PAGE STARTS HERE ───────────────────────────────

st.set_page_config("PetroPulse", "🛢️", layout="wide")
st.sidebar.title("Controls")

# Load articles
engine = get_engine()
df = get_articles_df(engine)
if df.empty:
    st.error("No articles in the database yet.")
    st.stop()

# Timestamps → pandas
df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
min_date = df["published_at"].dt.date.min()
max_date = df["published_at"].dt.date.max()

# Date filter UI
start_date = st.sidebar.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
end_date   = st.sidebar.date_input("End date",   max_date, min_value=min_date, max_value=max_date)
if start_date > end_date:
    st.sidebar.error("Start must be ≤ End")
    st.stop()

# Ensure there's always an 'entities' column
if "entities" not in df.columns:
    df["entities"] = [[] for _ in range(len(df))]

# Parse entities (no crash if extract_entities missing)
df["entities_parsed"] = df["entities"].apply(extract_entities)

# Filter by title keywords
KEYWORDS = ["oil","gas","energy","petroleum","carbon","renewable","offshore","fuel"]
df = df[df["title"].str.lower().apply(lambda t: any(k in t for k in KEYWORDS))]

# Source multiselect
all_sources = sorted(df["source"].dropna().unique())
sel_sources = st.sidebar.multiselect("Sources", all_sources, default=all_sources[:5])
if sel_sources:
    df = df[df["source"].isin(sel_sources)]

# Topic multiselect
all_topics = sorted({e for ents in df["entities_parsed"] for e in ents})
sel_topics = st.sidebar.multiselect("Topics", all_topics)
if sel_topics:
    df = df[df["entities_parsed"].apply(lambda ents: any(t in ents for t in sel_topics))]

# Final date slicing
df = df[df["published_at"].dt.date.between(start_date, end_date)]
if df.empty:
    st.warning("No articles found. Try broadening your date range or filters.")
    st.caption(f"Data spans {min_date} → {max_date}.")
    st.stop()

# ─── KPI ROW ────────────────────────────────────────
st.markdown("## PetroPulse Dashboard")
c1, c2, c3 = st.columns(3)
with c1:
    kpi_card("Articles", f"{len(df)}",              "linear-gradient(135deg,#1E3A8A,#3B82F6)")
with c2:
    avg_sent = df["sentiment"].astype(float).mean()
    kpi_card("Avg Sentiment", f"{avg_sent:.3f}",   "linear-gradient(135deg,#065F46,#10B981)")
with c3:
    kpi_card("Sources",      f"{df['source'].nunique()}", "linear-gradient(135deg,#B45309,#F59E0B)")

# ─── Latest Articles Table ─────────────────────────
st.markdown("### Latest Articles")
tbl = df.copy()
tbl["Published"] = tbl["published_at"].apply(to_edtz)
tbl["Sentiment"] = tbl["sentiment"].apply(
    lambda x: "Positive" if x>0.1 else ("Negative" if x< -0.1 else "Neutral")
)
tbl["Link"] = tbl["url"].apply(lambda u: f'<a href="{u}" target="_blank">Read</a>')
display = (
    tbl[["Published","source","title","Sentiment","Link"]]
    .rename(columns={"source":"Source","title":"Title"})
    .sort_values("Published", ascending=False)
)
st.write(display.to_html(escape=False, index=False), unsafe_allow_html=True)

# ─── Sentiment Trend ───────────────────────────────
if compute_daily_sentiment_index:
    idx_df = compute_daily_sentiment_index(df)
else:
    tmp = df.dropna(subset=["published_at"])
    tmp["date"] = tmp["published_at"].dt.date
    idx_df = tmp.groupby("date", as_index=False)["sentiment"].mean()
sentiment_chart(idx_df, "Daily Sentiment Trend")

# ─── Volume Over Time ──────────────────────────────
st.subheader("Article Volume Over Time")
vol = df.copy()
vol["date"] = vol["published_at"].dt.date
vol_chart = (
    alt.Chart(vol)
    .mark_bar()
    .encode(x="date:T", y="count()", tooltip=["date:T","count()"])
    .properties(height=200)
    .interactive()
)
st.altair_chart(vol_chart, use_container_width=True)

# ─── Word Cloud ────────────────────────────────────
st.subheader("Word Cloud of Titles")
all_titles = " ".join(df["title"].dropna())
wc = WordCloud(width=600, height=300, background_color="white").generate(all_titles)
fig, ax = plt.subplots(figsize=(6,3))
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

# ─── Download CSV ──────────────────────────────────
st.subheader("Download Filtered Data")
buf = StringIO()
df.to_csv(buf, index=False)
st.download_button("Download as CSV", data=buf.getvalue(), file_name="petropulse.csv", mime="text/csv")

# ─── Footer ────────────────────────────────────────
st.markdown(
    f"<div style='opacity:0.6; font-size:12px;'>"
    f"Data from {min_date} to {max_date} • Generated {datetime.now():%Y-%m-%d %H:%M}"
    "</div>",
    unsafe_allow_html=True
)

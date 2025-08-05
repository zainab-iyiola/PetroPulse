#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from io import StringIO
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Project imports
from src.storage.db import get_engine, get_articles_df
from src.models.sentiment import compute_daily_sentiment_index
from src.nlp.pipeline import extract_entities

EASTERN = ZoneInfo("US/Eastern")

def to_edtz(ts):
    if ts is None or (isinstance(ts, float) and np.isnan(ts)):
        return None
    ts = pd.to_datetime(ts, utc=True, errors="coerce")
    if ts is pd.NaT:
        return None
    return ts.tz_convert(EASTERN)

def fallback_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    if "published_at" not in tmp.columns or "sentiment" not in tmp.columns:
        return pd.DataFrame(columns=["date", "sentiment"])
    tmp["published_at"] = pd.to_datetime(tmp["published_at"], utc=True, errors="coerce")
    tmp = tmp.dropna(subset=["published_at"])
    tmp["date"] = tmp["published_at"].dt.date
    out = tmp.groupby("date", as_index=False)["sentiment"].mean(numeric_only=True)
    out.columns = ["date", "sentiment"]
    return out

def kpi_card(label: str, value: str, gradient: str):
    st.markdown(
        f"""
        <div style="
            background: {gradient};
            padding: 18px 20px;
            border-radius: 14px;
            color: #ffffff;
            box-shadow: 0 6px 18px rgba(0,0,0,.08);
        ">
            <div style="opacity:.85; font-size:14px;">{label}</div>
            <div style="font-size:28px; font-weight:700; margin-top:6px;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def sentiment_chart(idx: pd.DataFrame, title: str = "Sentiment Index"):
    if idx.empty:
        st.info("No sentiment data for this range.")
        return
    chart = (
        alt.Chart(idx)
        .mark_line(point=True)
        .encode(
            x="date:T",
            y=alt.Y("sentiment:Q", scale=alt.Scale(domain=(-1,1))),
            tooltip=["date:T","sentiment:Q"]
        )
        .properties(height=200)
        .interactive()
    )
    st.subheader(title)
    st.altair_chart(chart, use_container_width=True)

# --- Page setup ---
st.set_page_config(page_title="PetroPulse", page_icon="🛢️", layout="wide")
st.sidebar.title("PetroPulse")
st.sidebar.caption("Filters & Controls")

# --- Load & filter data ---
engine = get_engine()
df_all = get_articles_df(engine)

if df_all.empty:
    st.error("No articles in the database yet.")
    st.stop()

df_all["published_at"] = pd.to_datetime(df_all["published_at"], utc=True, errors="coerce")
min_date = df_all["published_at"].dt.date.min()
max_date = df_all["published_at"].dt.date.max()

# Sidebar date inputs with smart defaults
start_date = st.sidebar.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date)
end_date   = st.sidebar.date_input("End date",   value=max_date, min_value=min_date, max_value=max_date)
if start_date > end_date:
    st.sidebar.error("Start must be ≤ End")
    st.stop()

# Keyword filter
ENERGY_KEYWORDS = ["oil","gas","energy","petroleum","hydrogen","carbon",
                   "renewable","offshore","emission","turbine","fuel","exploration"]
def is_energy(text):
    t = str(text).lower()
    return any(k in t for k in ENERGY_KEYWORDS)
df_all = df_all[df_all["title"].apply(is_energy)]

# Ensure entities column
if "entities" not in df_all.columns:
    df_all["entities"] = [[] for _ in range(len(df_all))]
df_all["entities_parsed"] = df_all["entities"].apply(extract_entities)

# Source & Topic sidebar filters
sources = sorted(df_all["source"].dropna().unique())
selected_sources = st.sidebar.multiselect("Sources", sources, default=sources[:5])

all_topics = sorted({ent for row in df_all["entities_parsed"] for ent in row})
selected_topics = st.sidebar.multiselect("Topics", all_topics)

# Apply filters
mask = pd.Series(True, index=df_all.index)
mask &= df_all["published_at"].dt.date.between(start_date, end_date)
if selected_sources:
    mask &= df_all["source"].isin(selected_sources)
if selected_topics:
    mask &= df_all["entities_parsed"].apply(lambda ents: any(t in ents for t in selected_topics))
df = df_all[mask].copy()

# No-data friendly message
if df.empty:
    st.warning("No articles found. Try broadening the date range or loosening filters.")
    st.caption(f"Available data spans {min_date} → {max_date}.")
    st.stop()

# --- Header KPIs ---
st.markdown("## PetroPulse Dashboard")
c1,c2,c3 = st.columns(3)
with c1:
    kpi_card("Articles", f"{len(df)}",      "linear-gradient(135deg,#1D4ED8,#0EA5E9)")
with c2:
    avg_s = df["sentiment"].astype(float).mean()
    kpi_card("Avg Sentiment", f"{avg_s:.3f}", "linear-gradient(135deg,#0F766E,#10B981)")
with c3:
    kpi_card("Sources", f"{df['source'].nunique()}", "linear-gradient(135deg,#EA580C,#F59E0B)")

# --- Latest Articles table ---
st.markdown("### Latest Articles")
display = df.copy()
display["Published"] = display["published_at"].apply(to_edtz).dt.strftime("%Y-%m-%d %H:%M")
display["Sentiment"] = display["sentiment"].apply(
    lambda x: "Positive" if x>0.1 else ("Negative" if x< -0.1 else "Neutral")
)
display["Link"] = display["url"].apply(lambda u: f'<a href="{u}" target="_blank">Read</a>')
table = display[["Published","source","title","Sentiment","Link"]].rename(
    columns={"source":"Source","title":"Title"}
).sort_values("Published", ascending=False)
st.markdown(table.to_html(escape=False,index=False), unsafe_allow_html=True)

# --- Sentiment Trend ---
idx = compute_daily_sentiment_index(df) if compute_daily_sentiment_index else fallback_daily_sentiment(df)
sentiment_chart(idx, "Daily Sentiment Trend")

# --- Article Volume Time Series ---
st.subheader("Article Volume Over Time")
vol = df.copy()
vol["date"] = vol["published_at"].dt.date
chart_ts = (
    alt.Chart(vol)
    .mark_bar()
    .encode(
        x="date:T",
        y="count()",
        tooltip=["date:T","count()"]
    )
    .properties(height=200)
    .interactive()
)
st.altair_chart(chart_ts, use_container_width=True)

# --- Word Cloud of Titles ---
st.subheader("Word Cloud (Titles)")
all_text = " ".join(df["title"].dropna().tolist())
wc = WordCloud(width=600, height=300, background_color="white").generate(all_text)
fig, ax = plt.subplots(figsize=(6,3))
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

# --- Download CSV Button ---
st.subheader("Download Data")
csv_buf = StringIO()
df.to_csv(csv_buf, index=False)
st.download_button(
    "Download filtered articles as CSV",
    data=csv_buf.getvalue(),
    file_name="petropulse_articles.csv",
    mime="text/csv"
)

# --- Footer ---
st.markdown(
    "<div style='font-size:12px;opacity:0.6'>"
    f"Data from {min_date} to {max_date} | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    "</div>", unsafe_allow_html=True
)

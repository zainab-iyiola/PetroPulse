#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import ast
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Project imports
from src.storage.db import get_engine, get_articles_df

# Sentiment function
try:
    from src.models.sentiment import compute_daily_sentiment_index
    HAVE_SENT_FN = True
except Exception:
    HAVE_SENT_FN = False

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


def label_sentiment(polarity: float) -> str:
    if pd.isna(polarity):
        return "Neutral"
    if polarity > 0.1:
        return "Positive"
    if polarity < -0.1:
        return "Negative"
    return "Neutral"


def extract_entities(ents) -> list[str]:
    if isinstance(ents, (list, tuple)):
        return ents
    if isinstance(ents, str):
        try:
            ev = ast.literal_eval(ents)
            if isinstance(ev, list):
                return ev
        except Exception:
            pass
    return []


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


def sentiment_chart(idx: pd.DataFrame, title: str = "Sentiment Index (Selected Range)"):
    if idx.empty:
        st.info("No sentiment data for the selected range yet.")
        return
    chart = (
        alt.Chart(idx)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("sentiment:Q", title=None, scale=alt.Scale(domain=(-1, 1))),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("sentiment:Q", title="Sentiment", format=".3f"),
            ],
        )
        .properties(height=260)
        .interactive()
    )
    st.subheader(title)
    st.altair_chart(chart, use_container_width=True)

# Page config
st.set_page_config(page_title="PetroPulse", page_icon="🛢️", layout="wide")
st.sidebar.title("PetroPulse")
st.sidebar.caption("Controls")

# Load data
engine = get_engine()
df_all = get_articles_df(engine)

if not df_all.empty:
    df_all["published_at"] = pd.to_datetime(df_all["published_at"], utc=True, errors="coerce")
else:
    df_all = pd.DataFrame(columns=["source","title","url","published_at","sentiment","entities"])

# Smart default date range
min_date = (df_all["published_at"].min().date() if not df_all.empty else datetime.now(timezone.utc).date() - timedelta(days=30))
max_date = (df_all["published_at"].max().date() if not df_all.empty else datetime.now(timezone.utc).date())
start_date = st.sidebar.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date)
end_date   = st.sidebar.date_input("End date",   value=max_date, min_value=min_date, max_value=max_date)

# Energy filter
ENERGY_KEYWORDS = [
    "oil","gas","energy","petroleum","hydrogen","carbon",
    "renewable","offshore","emission","turbine","fuel","exploration"
]
def is_energy_related(text: str) -> bool:
    return any(k in str(text).lower() for k in ENERGY_KEYWORDS)
df_all = df_all[df_all["title"].apply(is_energy_related)]

# Parse entities
df_all["entities_parsed"] = df_all["entities"].apply(extract_entities)
fallback_entities = df_all["entities_parsed"].apply(len).sum() == 0

# Topic filter
if fallback_entities:
    topics = sorted(ENERGY_KEYWORDS)
else:
    topics = sorted({t for sub in df_all["entities_parsed"] for t in sub})
selected_topics = st.sidebar.multiselect("Topics", options=topics)

# Source filter (default all sources)
sources = sorted(df_all["source"].dropna().unique())
selected_sources = st.sidebar.multiselect(
    "Sources", options=sources, default=sources
)

# Apply filters
mask = pd.Series(True, index=df_all.index)
mask &= df_all["published_at"].dt.date.between(start_date, end_date)
if selected_sources:
    mask &= df_all["source"].isin(selected_sources)
if selected_topics:
    if fallback_entities:
        mask &= df_all["title"].apply(lambda text: any(topic.lower() in str(text).lower() for topic in selected_topics))
    else:
        mask &= df_all["entities_parsed"].apply(lambda lst: any(t in lst for t in selected_topics))

df = df_all.loc[mask].copy()

# Header & empty state
st.markdown("## PetroPulse")
st.caption("Real-time oil & gas news, sentiment, word clouds, and trends")
if df.empty:
    st.warning("No articles found. Try broadening the date range or fewer filters.")
    st.stop()

# Sentiment labeling
df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
df["sentiment_label"] = df["sentiment"].apply(label_sentiment)

# KPI cards
articles_count = len(df)
avg_sentiment  = df["sentiment"].mean()
unique_sources = df["source"].nunique()

c1, c2, c3 = st.columns([1,1,1])
with c1:
    kpi_card("Articles", f"{articles_count:,}", "linear-gradient(135deg, #1D4ED8 0%, #0EA5E9 100%)")
with c2:
    kpi_card("Avg Sentiment", f"{avg_sentiment:.3f}" if not np.isnan(avg_sentiment) else "—", "linear-gradient(135deg, #0F766E 0%, #10B981 100%)")
with c3:
    kpi_card("Sources", f"{unique_sources:,}", "linear-gradient(135deg, #EA580C 0%, #F59E0B 100%)")

# Latest Articles
st.markdown("### Latest Articles")
display_df = df.copy()

display_df["Published"] = display_df["published_at"].apply(to_edtz).dt.strftime("%Y-%m-%d %H:%M %Z")
display_df["Sentiment"] = display_df["sentiment_label"]
display_df["Link"] = display_df["url"].apply(lambda u: f'<a href="{u}" target="_blank">Read</a>' if isinstance(u, str) and u else "")

display_df = display_df.rename(columns={"source":"Source","title":"Title"})
display_df = display_df[["Published","Source","Title","Sentiment","Link"]].sort_values("Published", ascending=False)
st.markdown(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)

# Download CSV
csv = display_df.drop(columns=["Link"]).to_csv(index=False).encode("utf-8")
st.download_button("Download articles as CSV", data=csv, file_name="petropulse_articles.csv", mime="text/csv")

# Sentiment Over Time
if HAVE_SENT_FN:
    idx = compute_daily_sentiment_index(df.rename(columns={"published_at":"published_at"}))
else:
    idx = fallback_daily_sentiment(df)

sentiment_chart(idx)
st.markdown("<div style='opacity:.6; font-size:12px; margin-top:12px;'>Data shown in Eastern Time. Sentiment is average text polarity per day.</div>", unsafe_allow_html=True)

# Word Cloud
st.subheader("Word Cloud")
text = " ".join(df["title"].tolist())
wc = WordCloud(width=800, height=400, background_color="white").generate(text)
fig, ax = plt.subplots(figsize=(8,4))
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

# Article Volume Over Time
st.subheader("Article Volume Over Time")
vol = df.copy()
vol["date"] = vol["published_at"].dt.date
vol = vol.groupby("date").size().reset_index(name="count")
vol["date"] = pd.to_datetime(vol["date"])
chart_ts = (
    alt.Chart(vol)
    .mark_line(point=True)
    .encode(x="date:T", y="count:Q", tooltip=["date","count"])  
    .properties(height=300)
)
st.altair_chart(chart_ts, use_container_width=True)

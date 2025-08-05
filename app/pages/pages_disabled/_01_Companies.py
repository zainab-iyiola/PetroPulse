import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
from collections import Counter
from src.storage.db import get_engine, get_articles_df

st.set_page_config(layout="wide")
st.title("Companies & Places Mentioned")

engine = get_engine()
df = get_articles_df(engine)
if df.empty or "entities" not in df.columns:
    st.info("No entities yet. Try re-running ingest.")
    st.stop()

# Expand entities like "ORG:ExxonMobil;GPE:U.S.;ORG:BP"
def parse_entities(series: pd.Series, wanted_labels=("ORG", "GPE")) -> pd.DataFrame:
    items = []
    for s in series.dropna():
        for e in str(s).split(";"):
            if ":" in e:
                label, text = e.split(":", 1)
                if label in wanted_labels and text.strip():
                    items.append((label, text.strip()))
    return pd.DataFrame(items, columns=["label", "text"])

ents = parse_entities(df["entities"])
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top Organizations (ORG)")
    orgs = ents[ents["label"] == "ORG"]["text"].str.strip().str.title()
    org_counts = orgs.value_counts().head(20)
    if len(org_counts) > 0:
        st.bar_chart(org_counts)
    else:
        st.info("No organizations found.")

with col2:
    st.subheader("Top Places (GPE)")
    gpes = ents[ents["label"] == "GPE"]["text"].str.strip().str.title()
    gpe_counts = gpes.value_counts().head(20)
    if len(gpe_counts) > 0:
        st.bar_chart(gpe_counts)
    else:
        st.info("No places found.")

st.subheader("Articles mentioning selected entity")
entity = st.text_input("Type an organization/place to filter articles", "")
if entity:
    entity_lower = entity.lower()
    mask = df["entities"].fillna("").str.lower().str.contains(entity_lower)
    st.dataframe(df.loc[mask, ["published_at", "source", "title", "url"]].sort_values("published_at", ascending=False).head(50), use_container_width=True)

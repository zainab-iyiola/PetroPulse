import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # make src importable

import streamlit as st
import pandas as pd
import requests
import feedparser
from datetime import datetime, timezone

# Pull feed lists and HTTP defaults from your scraper
from src.scrapers.news_scraper import (
    DEFAULT_FEEDS,
    GOV_DATA, MAJOR_WIRES, INDUSTRY_CORE, LNG_SHIPPING, UTILS_RENEWABLES,
    COMPANY_SUPERMAJORS, COMPANY_NOC, COMPANY_MIDSTREAM_LNG, COMPANY_OFS_EPC,
    COMPANY_APAC_AUS, COMPANY_EU_INDEP,
    HEADERS, REQUEST_TIMEOUT,
)

# ---- Optional: group feeds for filtering ----
FEED_GROUPS = {
    "All": DEFAULT_FEEDS,
    "Gov / Data": GOV_DATA,
    "Major Wires": MAJOR_WIRES,
    "Industry Core": INDUSTRY_CORE,
    "LNG / Shipping": LNG_SHIPPING,
    "Utilities / Renewables": UTILS_RENEWABLES,
    "Company: Supermajors": COMPANY_SUPERMAJORS,
    "Company: NOCs": COMPANY_NOC,
    "Company: Midstream & LNG": COMPANY_MIDSTREAM_LNG,
    "Company: OFS / EPC": COMPANY_OFS_EPC,
    "Company: APAC / Australia": COMPANY_APAC_AUS,
    "Company: Europe Independents": COMPANY_EU_INDEP,
}

st.set_page_config(layout="wide")
st.title("Feeds Health")
st.caption("Ping each RSS/Atom feed → status, latency, items, and last publish time.")

# ---------- Controls ----------
col_a, col_b, col_c = st.columns([2,2,1])
with col_a:
    group = st.selectbox("Feed group", list(FEED_GROUPS.keys()))
with col_b:
    limit_per_feed = st.slider("Max items to parse per feed", 5, 50, 20, 5)
with col_c:
    ttl_min = st.selectbox("Cache (minutes)", [1, 5, 10, 30], index=1)

feeds = FEED_GROUPS[group]

@st.cache_data(show_spinner=False, ttl=lambda: ttl_min*60)
def check_feeds(feeds: list[str], limit: int = 20) -> pd.DataFrame:
    rows = []
    for url in feeds:
        t0 = time.time()
        status_code = None
        ok = False
        err = ""
        # 1) HEAD request is often blocked; go straight to GET
        try:
            r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            status_code = r.status_code
            ok = r.ok
        except Exception as e:
            err = f"request: {e}"

        latency_ms = int((time.time() - t0) * 1000)

        # 2) Try parsing entries
        entries = 0
        last_title = ""
        last_dt = None
        try:
            parsed = feedparser.parse(url)
            items = getattr(parsed, "entries", [])[:limit]
            entries = len(items)
            if entries > 0:
                e0 = items[0]
                last_title = e0.get("title", "")
                last_dt = _coerce_date(e0)
        except Exception as e:
            if err:
                err += f" | parse: {e}"
            else:
                err = f"parse: {e}"

        rows.append({
            "feed_url": url,
            "status_code": status_code,
            "ok": ok,
            "latency_ms": latency_ms,
            "entries": entries,
            "last_title": last_title,
            "last_published_utc": last_dt,
            "error": err,
        })

    df = pd.DataFrame(rows)
    # sort: failures first, then slowest
    df["ok_sort"] = df["ok"].fillna(False).astype(int)
    df = df.sort_values(["ok_sort", "latency_ms"], ascending=[True, True]).drop(columns=["ok_sort"])
    return df

def _coerce_date(entry) -> str | None:
    # Try multiple date fields, return ISO UTC string
    for key in ("published_parsed", "updated_parsed"):
        tup = getattr(entry, key, None)
        if tup:
            try:
                dt = datetime(tup.tm_year, tup.tm_mon, tup.tm_mday,
                              tup.tm_hour, tup.tm_min, tup.tm_sec, tzinfo=timezone.utc)
                return dt.isoformat()
            except Exception:
                pass
    for key in ("published", "updated"):
        val = entry.get(key)
        if val:
            try:
                tup = feedparser._parse_date(val)
                if tup:
                    dt = datetime(tup.tm_year, tup.tm_mon, tup.tm_mday,
                                  tup.tm_hour, tup.tm_min, tup.tm_sec, tzinfo=timezone.utc)
                    return dt.isoformat()
            except Exception:
                pass
    return None

# ---------- Run check ----------
if st.button("Run health check"):
    with st.spinner("Checking feeds…"):
        df = check_feeds(feeds, limit_per_feed)
    st.success(f"Checked {len(feeds)} feeds in current group.")
else:
    df = check_feeds(feeds, limit_per_feed)

# ---------- Display ----------
st.subheader("Results")
st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "feed_url": st.column_config.LinkColumn("Feed URL", display_text="Open"),
        "last_published_utc": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
    },
)

st.subheader("Quick stats")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Feeds OK", int(df["ok"].fillna(False).sum()))
with c2:
    st.metric("Avg latency (ms)", int(df["latency_ms"].mean() if len(df) else 0))
with c3:
    st.metric("Total entries parsed", int(df["entries"].sum()))

st.subheader("Entries by feed (top 20)")
top = df.sort_values("entries", ascending=False).head(20).set_index("feed_url")["entries"]
if len(top) > 0:
    st.bar_chart(top)
else:
    st.info("No entries parsed.")

# ---------- Export ----------
st.download_button(
    "⬇️ Download results (CSV)",
    data=df.to_csv(index=False),
    file_name="feeds_health.csv",
    mime="text/csv",
)

import feedparser
import requests
import pandas as pd
from newspaper import Article
from datetime import datetime, timezone
from urllib.parse import urlparse
from typing import List, Dict, Optional

# ------------------------------------------------------------
# This header tells websites who we are — helps avoid blocks
# ------------------------------------------------------------
REQUEST_TIMEOUT = 15
HEADERS = {"User-Agent": "PetroPulse/1.0 (+https://example.com)"}

# ------------------------------------------------------------
# RSS feed groups — each list represents a content category
# ------------------------------------------------------------

# Government and energy agency feeds
GOV_DATA = [
    "https://www.eia.gov/rss/todayinenergy.xml",
    "https://www.eia.gov/rss/whatsnew.xml",
    "https://www.iea.org/rss/news.xml",
    "https://www.iea.org/rss/pressreleases.xml",
]

# Major global news agencies
MAJOR_WIRES = [
    "https://www.reuters.com/markets/commodities/rss",
    "https://www.cnbc.com/id/10000113/device/rss/rss.html",
    "https://www.ft.com/companies/energy?format=rss",
    "https://www.ft.com/world?format=rss",
    "https://www.bbc.co.uk/news/business/rss.xml",
    "https://www.bbc.co.uk/news/world/rss.xml",
    "https://www.economist.com/business/rss.xml",
]

# Oil and gas industry core publications
INDUSTRY_CORE = [
    "https://www.ogj.com/__rss/rss2.xml",
    "https://www.worldoil.com/rss/",
    "https://www.rigzone.com/news/rss/",
    "https://www.offshore-energy.biz/feed/",
    "https://www.offshore-mag.com/__rss/rss2.xml",
    "https://www.jpt.spe.org/rss.xml",
    "https://pubs.spe.org/twa/rss",
    "https://www.upstreamonline.com/rss",
    "https://www.hartenergy.com/rss.xml",
    "https://www.energyvoice.com/feed/",
    "https://www.argusmedia.com/rss",
    "https://www.plattslive.com/feed",
]

# LNG and shipping sources
LNG_SHIPPING = [
    "https://www.lngindustry.com/rss",
    "https://splash247.com/feed/",
    "https://gcaptain.com/feed/",
    "https://www.offshorewind.biz/feed/",
]

# Utility and renewable energy news
UTILS_RENEWABLES = [
    "https://www.utilitydive.com/feeds/news/",
    "https://www.pv-tech.org/feed/",
    "https://www.windpowermonthly.com/rss",
]

# Supermajor oil & gas companies
COMPANY_SUPERMAJORS = [
    "https://corporate.exxonmobil.com/en/company/news/newsroom.rss",
    "https://www.shell.com/media/news-and-media-releases/_jcr_content/par/list.feed",
    "https://www.bp.com/en/global/corporate/news-and-insights/_jcr_content/par/section/section_0/teaser_list.feed",
    "https://www.totalenergies.com/media/news/newsroom/rss",
    "https://www.chevron.com/rss/press-releases",
    "https://www.eni.com/en-IT/media/press-releases.rss",
    "https://www.equinor.com/news.rss",
    "https://www.conocophillips.com/newsroom/rss/",
    "https://www.oxy.com/newsroom/press-releases/feed/",
]

# National Oil Companies and other large producers
COMPANY_NOC = [
    "https://www.saudiaramco.com/en/news/rss",
    "https://www.adnoc.ae/media-center/rss",
    "https://www.petrobras.com.br/en/press-center/news/feed/",
    "https://www.qatarenergy.qa/en/MediaCenter/Pages/press-releases.aspx?rss=1",
    "https://www.petronas.com/rss.xml",
    "https://www.repsol.com/en/press-room/_jcr_content.feed",
    "https://www.omv.com/en/news/press-releases?format=rss",
    "https://www.ecopetrol.com.co/wps/portal/ecopetrol-web/rss/noticias",
    "https://www.pemex.com/en/rss/news.xml",
]

# Midstream, LNG export, and pipeline operators
COMPANY_MIDSTREAM_LNG = [
    "https://www.enbridge.com/media-center/rss",
    "https://www.williams.com/feed/",
    "https://www.kindermorgan.com/rss/press-releases",
    "https://www.tcenergy.com/siteassets/rss/news.xml",
    "https://www.cheniere.com/rss/press-releases.xml",
    "https://semprainfrastructure.com/feed/",
    "https://www.tellurianinc.com/feed/",
]

# Oilfield services and EPC companies
COMPANY_OFS_EPC = [
    "https://investors.slb.com/news-releases/rss",
    "https://ir.halliburton.com/rss/press-releases.xml",
    "https://investors.bakerhughes.com/news-releases/rss",
    "https://www.saipem.com/en/media/press-releases/feed",
    "https://www.woodplc.com/news/rss",
    "https://www.mcddermott.com/rss/news.xml",
    "https://www.subsea7.com/en/media/press-releases/_jcr_content.feed",
    "https://www.technipFMC.com/media/press-releases/_jcr_content.feed",
]

# Asia-Pacific and Australia operators
COMPANY_APAC_AUS = [
    "https://www.woodside.com/newsroom/_jcr_content.feed",
    "https://www.santos.com/news/_jcr_content.feed",
    "https://www.originenergy.com.au/blog/feed/",
    "https://www.inpex.co.jp/english/news/feed/",
    "https://www.cnooc.com.cn/data/rss/englishNews.xml",
]

# European independents
COMPANY_EU_INDEP = [
    "https://www.varenergi.no/feed/",
    "https://www.harbourenergy.com/media-centre/_jcr_content.feed",
    "https://www.neptunenergy.com/newsroom/_jcr_content.feed",
    "https://www.tullowoil.com/media-feed.xml",
    "https://www.kosmosenergy.com/feed/",
]

# Combine all feeds into one master list
DEFAULT_FEEDS = (
    GOV_DATA
    + MAJOR_WIRES
    + INDUSTRY_CORE
    + LNG_SHIPPING
    + UTILS_RENEWABLES
    + COMPANY_SUPERMAJORS
    + COMPANY_NOC
    + COMPANY_MIDSTREAM_LNG
    + COMPANY_OFS_EPC
    + COMPANY_APAC_AUS
    + COMPANY_EU_INDEP
)

# ------------------------------------------------------------
# Parses a date from the feed entry — falls back to current time
# ------------------------------------------------------------
def _parse_date(entry) -> datetime:
    for key in ("published_parsed", "updated_parsed"):
        tup = getattr(entry, key, None)
        if tup:
            try:
                return datetime(
                    tup.tm_year, tup.tm_mon, tup.tm_mday,
                    tup.tm_hour, tup.tm_min, tup.tm_sec, tzinfo=timezone.utc
                )
            except Exception:
                pass
    for key in ("published", "updated"):
        val = entry.get(key)
        if val:
            try:
                tup = feedparser._parse_date(val)
                if tup:
                    return datetime(
                        tup.tm_year, tup.tm_mon, tup.tm_mday,
                        tup.tm_hour, tup.tm_min, tup.tm_sec, tzinfo=timezone.utc
                    )
            except Exception:
                pass
    return datetime.now(timezone.utc)

# ------------------------------------------------------------
# Extracts a source name if title is missing — uses domain name
# ------------------------------------------------------------
def _get_source_name(parsed, fallback_url: str) -> str:
    if getattr(parsed, "feed", None) and "title" in parsed.feed:
        return parsed.feed["title"]
    try:
        netloc = urlparse(fallback_url).netloc
        return netloc.replace("www.", "")
    except Exception:
        return "Unknown Source"

# ------------------------------------------------------------
# Fetches articles from all feeds — returns list of dicts
# ------------------------------------------------------------
def fetch_rss_entries(feeds: Optional[List[str]] = None, limit_per_feed: int = 30) -> List[Dict]:
    feeds = feeds or DEFAULT_FEEDS
    rows: List[Dict] = []

    for url in feeds:
        try:
            parsed = feedparser.parse(url)
            feed_title = _get_source_name(parsed, url)

            for entry in getattr(parsed, "entries", [])[:limit_per_feed]:
                dt = _parse_date(entry)
                rows.append({
                    "source": feed_title,
                    "title": entry.get("title", "") or "",
                    "url": entry.get("link", "") or "",
                    "published_at": dt
                })

        except Exception as e:
            print(f"[fetch_rss_entries] {url} -> {e}")
            continue

    # Convert to DataFrame so we can clean things up
    if not rows:
        return []

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["url", "source"]).drop_duplicates(subset=["url"])

    # Print a breakdown of how many articles were collected from each source
    print("[fetch_rss_entries] Sources collected:\n", df["source"].value_counts())

    return df.to_dict(orient="records")

# ------------------------------------------------------------
# Tries to extract the full article text from a URL
# ------------------------------------------------------------
def fetch_article_text(url: str) -> str:
    try:
        # First try: download the page manually and parse with newspaper3k
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        art = Article(url)
        art.set_html(resp.text)
        art.parse()
        if art.text and len(art.text.strip()) > 0:
            return art.text

    except Exception as e:
        print(f"[fetch_article_text:req] {url} -> {e}")

    try:
        # Fallback: let newspaper3k handle everything
        art2 = Article(url)
        art2.download()
        art2.parse()
        if art2.text and len(art2.text.strip()) > 0:
            return art2.text
    except Exception as e:
        print(f"[fetch_article_text:newspaper] {url} -> {e}")

    try:
        # If everything fails, return part of the raw HTML so we have something
        resp2 = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        return resp2.text[:5000]
    except Exception:
        return ""

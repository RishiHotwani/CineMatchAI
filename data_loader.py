# ============================================================
#  CineMatch AI — Data Loader  (Fast Edition)
#
#  Key optimisations:
#  • ThreadPoolExecutor — 10 parallel OMDb requests (10× faster)
#  • No time.sleep()  — OMDb free tier is 1000/day not per-second
#  • MovieLens with chunked streaming + real MB progress bar
#  • Poster cache loaded once at import; persisted every 20 entries
# ============================================================

import os, io, re, json, zipfile, hashlib, urllib.parse, urllib.request
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import numpy as np
import streamlit as st

from config import (
    OMDB_API_KEY, OMDB_BASE_URL,
    OMDB_SEED_MOVIES, MOVIELENS_URL, CACHE_DIR,
)

os.makedirs(CACHE_DIR, exist_ok=True)
_lock = threading.Lock()   # protects shared state in worker threads

# ─────────────────────────────────────────────
#  OMDb core
# ─────────────────────────────────────────────

def _omdb(params: dict) -> dict:
    params["apikey"] = OMDB_API_KEY
    try:
        r = requests.get(OMDB_BASE_URL, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        return data if data.get("Response") == "True" else {}
    except Exception:
        return {}


def _fetch_one(title: str) -> dict | None:
    data = _omdb({"t": title, "plot": "full", "type": "movie"})
    if not data:
        return None
    imdb_rating, rt_rating = 0.0, ""
    for rat in data.get("Ratings", []):
        if rat["Source"] == "Internet Movie Database":
            try: imdb_rating = float(rat["Value"].split("/")[0])
            except: pass
        if rat["Source"] == "Rotten Tomatoes":
            rt_rating = rat["Value"]
    poster = data.get("Poster", "")
    if poster == "N/A": poster = ""
    return {
        "imdb_id":     data.get("imdbID", ""),
        "title":       data.get("Title", title),
        "year":        data.get("Year", ""),
        "rated":       data.get("Rated", ""),
        "runtime":     data.get("Runtime", ""),
        "genre":       data.get("Genre", ""),
        "director":    data.get("Director", ""),
        "writer":      data.get("Writer", ""),
        "actors":      data.get("Actors", ""),
        "plot":        data.get("Plot", ""),
        "language":    data.get("Language", ""),
        "country":     data.get("Country", ""),
        "awards":      data.get("Awards", ""),
        "poster":      poster,
        "imdb_rating": imdb_rating,
        "imdb_votes":  data.get("imdbVotes", "0").replace(",", ""),
        "rt_rating":   rt_rating,
        "box_office":  data.get("BoxOffice", ""),
        "production":  data.get("Production", ""),
    }


def _make_soup(row: pd.Series) -> str:
    parts = [str(row.get("plot", "") or "")]
    g = str(row.get("genre",    "") or ""); parts += [g, g]
    d = str(row.get("director", "") or ""); parts += [d, d]
    parts += [str(row.get(k, "") or "") for k in ("actors","writer","country","language")]
    return " ".join(parts).lower()


# ─────────────────────────────────────────────
#  fetch_omdb_movies  — PARALLEL
# ─────────────────────────────────────────────

def fetch_omdb_movies() -> pd.DataFrame:
    cache_path = os.path.join(CACHE_DIR, "omdb_movies.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    titles = list(dict.fromkeys(OMDB_SEED_MOVIES))
    total  = len(titles)

    st.markdown("""
    <div style='background:rgba(0,245,212,.05);border:1px solid rgba(0,245,212,.2);
                border-radius:10px;padding:.9rem 1.2rem;margin-bottom:.8rem;font-size:.87rem;color:#94a3b8'>
      🎬 <b style='color:#00f5d4'>First-time setup</b> — fetching movie catalogue from OMDb in parallel.
      This runs <b>once only</b>; all future launches load from local cache instantly.
    </div>
    """, unsafe_allow_html=True)

    bar   = st.progress(0.0, text=f"⚡ 0 / {total} movies fetched…")
    done  = [0]
    rows_out, seen = [], set()

    def _worker(t: str):
        row = _fetch_one(t)
        with _lock:
            done[0] += 1
            bar.progress(done[0] / total,
                         text=f"⚡ {done[0]} / {total} — {t[:45]}")
        return row

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_worker, t): t for t in titles}
        for fut in as_completed(futures):
            row = fut.result()
            if row and row.get("imdb_id") and row["title"] not in seen:
                seen.add(row["title"])
                rows_out.append(row)

    bar.empty()
    df = pd.DataFrame(rows_out)
    df["imdb_rating"] = pd.to_numeric(df["imdb_rating"], errors="coerce").fillna(0)
    df["soup"] = df.apply(_make_soup, axis=1)
    df = df.drop_duplicates(subset="imdb_id").reset_index(drop=True)
    df.to_parquet(cache_path, index=False)
    st.success(f"✅ {len(df)} movies loaded & cached — next launch will be instant!")
    return df


# ─────────────────────────────────────────────
#  fetch_movielens  — chunked download
# ─────────────────────────────────────────────

def fetch_movielens() -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings_path = os.path.join(CACHE_DIR, "ml100k_ratings.parquet")
    movies_path  = os.path.join(CACHE_DIR, "ml100k_movies.parquet")
    if os.path.exists(ratings_path) and os.path.exists(movies_path):
        return pd.read_parquet(ratings_path), pd.read_parquet(movies_path)

    st.markdown("""
    <div style='background:rgba(255,193,7,.05);border:1px solid rgba(255,193,7,.2);
                border-radius:10px;padding:.9rem 1.2rem;margin-bottom:.8rem;font-size:.87rem;color:#94a3b8'>
      📦 <b style='color:#ffc107'>Downloading MovieLens 100K</b> (~5 MB).
      One-time download — cached forever after this.
    </div>
    """, unsafe_allow_html=True)

    bar = st.progress(0.0, text="⬇️ Connecting…")
    try:
        resp       = urllib.request.urlopen(MOVIELENS_URL)
        total_size = int(resp.headers.get("Content-Length", 5_000_000))
        buf, downloaded = io.BytesIO(), 0
        while True:
            chunk = resp.read(65_536)
            if not chunk: break
            buf.write(chunk)
            downloaded += len(chunk)
            pct = min(downloaded / total_size, 1.0)
            bar.progress(pct, text=f"⬇️ {downloaded/1_048_576:.1f} / {total_size/1_048_576:.1f} MB")
        bar.progress(1.0, text="📦 Extracting…")
        buf.seek(0)
        zf = zipfile.ZipFile(buf)
    except Exception as e:
        bar.empty()
        st.error(f"❌ Download failed: {e}")
        st.stop()

    with zf.open("ml-100k/u.data") as f:
        ratings = pd.read_csv(f, sep="\t",
                              names=["user_id","item_id","rating","timestamp"])
    with zf.open("ml-100k/u.item") as f:
        movies = pd.read_csv(f, sep="|", encoding="latin-1",
                             names=["item_id","title"]+[f"f{i}" for i in range(22)],
                             usecols=["item_id","title"])

    ratings.to_parquet(ratings_path, index=False)
    movies.to_parquet(movies_path,   index=False)
    bar.empty()
    st.success(f"✅ MovieLens loaded — {len(ratings):,} ratings from {ratings['user_id'].nunique()} users!")
    return ratings, movies


# ─────────────────────────────────────────────
#  Poster helpers
# ─────────────────────────────────────────────

_POSTER_CACHE_PATH = os.path.join(CACHE_DIR, "poster_cache.json")
_poster_cache: dict[str, str] = {}

def _load_poster_cache():
    global _poster_cache
    if os.path.exists(_POSTER_CACHE_PATH):
        try: _poster_cache = json.loads(open(_POSTER_CACHE_PATH).read())
        except: _poster_cache = {}

def _save_poster_cache():
    try: open(_POSTER_CACHE_PATH, "w").write(json.dumps(_poster_cache))
    except: pass

_load_poster_cache()

_SVG_PALETTE = [
    ("#0e1320","#00f5d4"),("#120d22","#e94560"),
    ("#0d1117","#ffb703"),("#0b1622","#7b2fff"),
    ("#160e14","#ff6b6b"),("#0a1628","#00d4ff"),
]

def _svg_card(title: str) -> str:
    idx = int(hashlib.md5(title.encode()).hexdigest(), 16) % len(_SVG_PALETTE)
    bg, accent = _SVG_PALETTE[idx]
    words, lines, cur = title.split(), [], ""
    for w in words:
        if len(cur)+len(w)+1 > 16 and cur:
            lines.append(cur.strip()); cur = w
        else: cur += " "+w
    if cur.strip(): lines.append(cur.strip())
    lines = lines[:4]
    y0 = 210 - len(lines)*18
    txts = "".join(
        f'<text x="150" y="{y0+i*34}" font-family="Georgia,serif" '
        f'font-size="20" fill="{accent}" text-anchor="middle" font-weight="bold">{ln}</text>'
        for i, ln in enumerate(lines)
    )
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="300" height="450">'
           f'<rect width="300" height="450" fill="{bg}"/>'
           f'<rect x="0" y="0" width="300" height="3" fill="{accent}"/>'
           f'<rect x="0" y="447" width="300" height="3" fill="{accent}"/>'
           f'<text x="150" y="130" font-size="60" text-anchor="middle" fill="{accent}" opacity="0.12">🎬</text>'
           f'{txts}'
           f'<text x="150" y="420" font-family="Georgia,serif" font-size="10" '
           f'fill="{accent}" text-anchor="middle" opacity="0.35">CINEMATCH AI</text></svg>')
    return "data:image/svg+xml," + urllib.parse.quote(svg)


def _clean_ml_title(raw: str) -> str:
    title = re.sub(r'\s*\(\d{4}\)\s*$', '', raw).strip()
    m = re.match(r'^(.+),\s*(The|A|An)\s*$', title, re.IGNORECASE)
    if m: title = f"{m.group(2)} {m.group(1)}"
    return title


@st.cache_data(show_spinner=False, ttl=86400)
def get_poster(raw_title: str) -> str:
    if raw_title in _poster_cache:
        return _poster_cache[raw_title]
    clean  = _clean_ml_title(raw_title)
    data   = _omdb({"t": clean, "type": "movie"})
    poster = data.get("Poster", "")
    url    = poster if (poster and poster != "N/A") else _svg_card(clean)
    _poster_cache[raw_title] = url
    if len(_poster_cache) % 20 == 0:
        _save_poster_cache()
    return url


def poster_from_row(row) -> str:
    url = str(row.get("poster") or "")
    if url and url.startswith("http"):
        return url
    return get_poster(str(row.get("title", "")))

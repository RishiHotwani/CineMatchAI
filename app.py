# ============================================================
#  CineMatch AI — Main Streamlit Application
#  OMDb-only · Dark cinema theme · Neon accents
#  Mobile-responsive: fluid grid, stacked layouts, touch UI
# ============================================================

import streamlit as st
import pandas as pd

from config import TOP_N
from data_loader import fetch_omdb_movies, fetch_movielens, get_poster, poster_from_row
from recommenders import ContentRecommender, CollabRecommender, HybridRecommender

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",   # collapsed by default on mobile
)

# ── Global CSS  (desktop-first + mobile media queries) ────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@500&display=swap');

/* ═══════════════════════════════
   DESIGN TOKENS
═══════════════════════════════ */
:root {
    --bg:       #07090f;
    --bg-card:  #0d1119;
    --bg-mid:   #111827;
    --border:   rgba(255,255,255,0.07);
    --teal:     #00f5d4;
    --amber:    #ffc107;
    --rose:     #f43f5e;
    --violet:   #818cf8;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --r:        10px;
    --gap:      .75rem;
}

/* ═══════════════════════════════
   BASE RESET
═══════════════════════════════ */
html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080b14, #050709) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
#MainMenu, footer, header { visibility: hidden; }

/* Make main content area use full width on mobile */
[data-testid="stAppViewContainer"] > .main > .block-container {
    padding: 1rem 1rem 2rem !important;
    max-width: 100% !important;
}

/* ═══════════════════════════════
   HERO BANNER
═══════════════════════════════ */
.hero {
    position: relative; overflow: hidden;
    background: linear-gradient(135deg, #0d1119 0%, #0f0d1e 50%, #0d1119 100%);
    border: 1px solid var(--border); border-radius: var(--r);
    padding: 2.8rem 3.5rem; margin-bottom: 1.8rem;
}
.hero::before {
    content: ''; position: absolute; inset: 0; pointer-events: none;
    background:
        radial-gradient(ellipse at 5% 50%,   rgba(0,245,212,.08) 0%, transparent 55%),
        radial-gradient(ellipse at 95% 20%,  rgba(244,63,94,.06)  0%, transparent 55%),
        radial-gradient(ellipse at 50% 100%, rgba(129,140,248,.04) 0%, transparent 50%);
}
.hero-logo {
    font-family: 'Bebas Neue', sans-serif; font-size: 4.2rem;
    letter-spacing: .06em; line-height: 1; margin: 0 0 .3rem;
    background: linear-gradient(100deg, var(--teal) 0%, var(--amber) 50%, var(--rose) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { color: var(--muted); font-size: .9rem; letter-spacing: .04em; margin: 0; }
.hero-badges { display: flex; gap: .6rem; margin-top: 1rem; flex-wrap: wrap; }
.hbadge {
    font-family: 'JetBrains Mono', monospace; font-size: .7rem;
    padding: .3rem .8rem; border-radius: 999px; border: 1px solid; letter-spacing: .05em;
}
.hbadge-t { color:var(--teal);   border-color:rgba(0,245,212,.3);   background:rgba(0,245,212,.06); }
.hbadge-a { color:var(--amber);  border-color:rgba(255,193,7,.3);    background:rgba(255,193,7,.06); }
.hbadge-r { color:var(--rose);   border-color:rgba(244,63,94,.3);    background:rgba(244,63,94,.06); }
.hbadge-v { color:var(--violet); border-color:rgba(129,140,248,.3);  background:rgba(129,140,248,.06); }

/* ═══════════════════════════════
   SECTION HEADERS
═══════════════════════════════ */
.sh {
    font-family: 'Bebas Neue', sans-serif; font-size: 1.85rem;
    letter-spacing: .07em; margin: 1.4rem 0 1rem; color: var(--text);
}
.sh span      { color: var(--teal);  }
.sh-amber span { color: var(--amber); }
.sh-rose  span { color: var(--rose);  }

/* ═══════════════════════════════
   INFO BOXES
═══════════════════════════════ */
.infobox {
    background: rgba(0,245,212,.04); border: 1px solid rgba(0,245,212,.18);
    border-radius: var(--r); padding: .9rem 1.2rem; margin-bottom: 1.2rem;
    font-size: .87rem; color: var(--muted); line-height: 1.65;
}
.infobox b { color: var(--teal); }
.infobox-amber { background:rgba(255,193,7,.04);  border-color:rgba(255,193,7,.18); }
.infobox-amber b { color: var(--amber); }
.infobox-rose  { background:rgba(244,63,94,.04);  border-color:rgba(244,63,94,.18); }
.infobox-rose b  { color: var(--rose); }

/* ═══════════════════════════════
   MOVIE CARD  (fluid grid item)
═══════════════════════════════ */
.card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--r); overflow: hidden; position: relative;
    transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
    height: 100%;
}
/* hover only on real pointer devices (not touch) */
@media (hover: hover) {
    .card:hover {
        transform: translateY(-5px) scale(1.015);
        box-shadow: 0 18px 40px rgba(0,0,0,.55), 0 0 24px rgba(0,245,212,.07);
        border-color: rgba(0,245,212,.35);
    }
}
.card img { width: 100%; aspect-ratio: 2/3; object-fit: cover; display: block; }
.card-body { padding: .85rem .9rem; }
.card-title {
    font-weight: 600; font-size: .93rem; color: var(--text);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin: 0 0 .4rem;
}
.card-meta { display: flex; gap: .4rem; flex-wrap: wrap; align-items: center; font-size: .76rem; }
.pill { padding: .18rem .5rem; border-radius: 4px; font-size: .72rem; font-weight: 500; }
.pill-teal   { background:rgba(0,245,212,.12);  color:var(--teal);   }
.pill-amber  { background:rgba(255,193,7,.12);   color:var(--amber);  }
.pill-rose   { background:rgba(244,63,94,.12);   color:var(--rose);   }
.pill-violet { background:rgba(129,140,248,.12); color:var(--violet); }
.pill-muted  { background:rgba(255,255,255,.06); color:var(--muted);  }
.score-badge {
    position: absolute; top: .55rem; right: .55rem;
    font-family: 'JetBrains Mono', monospace; font-size: .69rem; font-weight: 500;
    padding: .22rem .55rem; border-radius: 999px; backdrop-filter: blur(6px);
    border: 1px solid rgba(255,255,255,.12);
}
.score-c { background:rgba(0,245,212,.82); color:#000; }
.score-l { background:rgba(255,193,7,.88); color:#000; }
.score-h { background:rgba(244,63,94,.88); color:#fff; }

/* ═══════════════════════════════
   RESPONSIVE CARD GRID
   Uses CSS grid so it naturally
   reflows on any screen width.
═══════════════════════════════ */
.card-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);  /* 5 cols on desktop */
    gap: var(--gap);
    margin-bottom: 1.5rem;
}

/* ═══════════════════════════════
   SELECTED MOVIE HERO CARD
═══════════════════════════════ */
.movie-hero {
    display: flex; gap: 2rem; padding: 1.5rem;
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--r); margin-bottom: 1.5rem;
    align-items: flex-start;
}
.movie-hero img {
    width: 160px; border-radius: 8px; flex-shrink: 0;
    object-fit: cover; aspect-ratio: 2/3;
}
.movie-hero-info h2 {
    font-family: 'Bebas Neue', sans-serif; font-size: 2.2rem;
    letter-spacing: .05em; margin: 0 0 .5rem; color: var(--text);
}

/* ═══════════════════════════════
   USER HISTORY CARD
═══════════════════════════════ */
.user-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--r); padding: 1.2rem 1.5rem; margin-bottom: 1.2rem;
}
.user-card-title {
    font-family: 'Bebas Neue', sans-serif; font-size: 1.3rem;
    letter-spacing: .06em; color: var(--amber); margin: 0 0 .8rem;
}

/* ═══════════════════════════════
   STAT CHIPS
═══════════════════════════════ */
.stat-chip {
    display: inline-block; background: var(--bg-mid); border: 1px solid var(--border);
    border-radius: var(--r); padding: .7rem 1.2rem; text-align: center; margin: .25rem;
}
.stat-chip .val {
    font-family: 'Bebas Neue', sans-serif; font-size: 1.8rem;
    color: var(--teal); line-height: 1;
}
.stat-chip .lbl { font-size: .7rem; color: var(--muted); text-transform: uppercase; letter-spacing: .09em; }

/* ═══════════════════════════════
   SIDEBAR
═══════════════════════════════ */
.sb-logo {
    font-family: 'Bebas Neue', sans-serif; font-size: 1.9rem; letter-spacing: .05em;
    background: linear-gradient(90deg, var(--teal), var(--amber));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: .2rem;
}
.sb-tag { text-align: center; font-size: .7rem; color: var(--muted); letter-spacing: .12em; margin-bottom: 1.2rem; }
.sb-desc { font-size: .78rem; color: var(--muted); line-height: 1.65; }
.sb-desc b.t { color: var(--teal);  }
.sb-desc b.a { color: var(--amber); }
.sb-desc b.r { color: var(--rose);  }

/* ═══════════════════════════════
   STREAMLIT WIDGET OVERRIDES
═══════════════════════════════ */
div[data-baseweb="select"] > div {
    background: var(--bg-card) !important; border-color: var(--border) !important;
    color: var(--text) !important; border-radius: var(--r) !important;
}
div[data-testid="stRadio"] > div { display: flex; gap: .5rem; flex-wrap: wrap; }
div[data-testid="stRadio"] > div label {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: 999px !important; padding: .4rem 1.1rem !important;
    color: var(--muted) !important; font-size: .83rem !important;
    cursor: pointer; transition: all .18s;
    /* larger tap target on mobile */
    min-height: 40px; display: flex; align-items: center;
}
div[data-testid="stRadio"] > div label:has(input:checked) {
    background: linear-gradient(90deg, var(--teal), #00b4d8) !important;
    border-color: transparent !important; color: #000 !important; font-weight: 700 !important;
}
/* Bigger touch targets for selects on mobile */
div[data-baseweb="select"] { min-height: 48px; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }


/* ═══════════════════════════════════════════════════════
   ██████  MOBILE BREAKPOINTS  ██████
   Everything below overrides the desktop styles above.
═══════════════════════════════════════════════════════ */

/* ── Tablet: ≤ 1024px ── */
@media (max-width: 1024px) {
    .card-grid { grid-template-columns: repeat(3, 1fr); }
    .hero { padding: 2rem 2rem; }
    .hero-logo { font-size: 3.2rem; }
}

/* ── Mobile landscape / small tablet: ≤ 768px ── */
@media (max-width: 768px) {
    /* Container padding */
    [data-testid="stAppViewContainer"] > .main > .block-container {
        padding: .75rem .75rem 2rem !important;
    }

    /* Hero — compact */
    .hero { padding: 1.4rem 1.2rem; margin-bottom: 1.2rem; }
    .hero-logo { font-size: 2.6rem; letter-spacing: .03em; }
    .hero-sub { font-size: .78rem; }
    .hero-badges { gap: .4rem; margin-top: .6rem; }
    .hbadge { font-size: .62rem; padding: .25rem .6rem; }

    /* Section headers */
    .sh { font-size: 1.45rem; margin: 1rem 0 .7rem; }

    /* Card grid: 2 columns on mobile */
    .card-grid { grid-template-columns: repeat(2, 1fr); gap: .5rem; }

    /* Movie hero: stack vertically */
    .movie-hero {
        flex-direction: column; gap: 1rem; padding: 1rem;
    }
    .movie-hero img { width: 100%; max-width: 200px; align-self: center; }
    .movie-hero-info h2 { font-size: 1.6rem; }

    /* User card */
    .user-card { padding: .9rem 1rem; }

    /* Stat chips: full row */
    .stat-chip { padding: .5rem .8rem; margin: .15rem; }
    .stat-chip .val { font-size: 1.4rem; }

    /* Info box */
    .infobox { font-size: .82rem; padding: .75rem 1rem; }

    /* Sidebar collapse button — bigger tap target */
    [data-testid="collapsedControl"] { width: 44px !important; height: 44px !important; }
}

/* ── Mobile portrait: ≤ 480px ── */
@media (max-width: 480px) {
    .hero { padding: 1.1rem 1rem; }
    .hero-logo { font-size: 2.1rem; }
    .hero-sub { font-size: .72rem; display: none; }  /* hide tagline on very small screens */
    .hero-badges { display: none; }                  /* hide tech badges; save space */

    .sh { font-size: 1.25rem; }

    /* Single column on very small phones */
    .card-grid { grid-template-columns: repeat(2, 1fr); gap: .4rem; }

    .card-title { font-size: .82rem; }
    .card-body { padding: .6rem .65rem; }
    .pill { font-size: .65rem; padding: .12rem .35rem; }
    .score-badge { font-size: .6rem; padding: .17rem .4rem; top: .4rem; right: .4rem; }

    .movie-hero-info h2 { font-size: 1.35rem; }

    /* Stack Streamlit columns on mobile using CSS override */
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  Load data + build models  (cached across reruns)
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_all():
    with st.spinner("🎬 Loading OMDb movie catalogue…"):
        omdb_df = fetch_omdb_movies()
    with st.spinner("📦 Loading MovieLens 100K ratings…"):
        ratings_df, ml_movies_df = fetch_movielens()
    with st.spinner("🧠 Building TF-IDF Content Engine…"):
        content_rec = ContentRecommender(omdb_df)
    with st.spinner("🔢 Training SVD Collaborative Filter…"):
        collab_rec = CollabRecommender(ratings_df, ml_movies_df)
    hybrid_rec = HybridRecommender(content_rec, collab_rec)
    return omdb_df, ratings_df, ml_movies_df, content_rec, collab_rec, hybrid_rec


omdb_df, ratings_df, ml_movies_df, content_rec, collab_rec, hybrid_rec = load_all()


# ─────────────────────────────────────────────────────────────
#  UI helpers
# ─────────────────────────────────────────────────────────────

def _stars(rating: float, out_of: float = 10.0) -> str:
    v = (rating / out_of) * 5
    full = int(v); frac = v - full; empty = 5 - full - (1 if frac >= 0.4 else 0)
    return (
        f'<span style="color:var(--amber);font-size:.85rem">'
        f'{"★"*full}{"½" if frac>=0.4 else ""}{"☆"*empty}'
        f'</span>'
    )


def _genre_pills(genre_str: str) -> str:
    colors = ["pill-teal", "pill-amber", "pill-rose"]
    genres = [g.strip() for g in str(genre_str or "").split(",") if g.strip()]
    return " ".join(
        f'<span class="pill {colors[i%3]}">{g}</span>'
        for i, g in enumerate(genres[:3])
    )


def _card_html(row: pd.Series, score: float, score_cls: str) -> str:
    title  = str(row.get("title", ""))
    poster = poster_from_row(row) if "poster" in row.index else get_poster(title)
    year   = str(row.get("year",  "") or "")[:4]
    genre  = str(row.get("genre", "") or "")
    rating = float(row.get("imdb_rating", 0) or 0)
    return f"""
<div class="card">
  <div class="score-badge {score_cls}">{score:.2f}</div>
  <img src="{poster}" alt="{title}" onerror="this.onerror=null;this.src=''"/>
  <div class="card-body">
    <div class="card-title" title="{title}">{title}</div>
    <div class="card-meta">
      {_stars(rating)}
      <span class="pill pill-muted">{year}</span>
      {_genre_pills(genre)}
    </div>
  </div>
</div>"""


def _ml_card_html(row: pd.Series, score: float) -> str:
    title  = str(row.get("title", ""))
    poster = get_poster(title)
    return f"""
<div class="card">
  <div class="score-badge score-l">{score:.2f}★</div>
  <img src="{poster}" alt="{title}" onerror="this.onerror=null;this.src=''"/>
  <div class="card-body">
    <div class="card-title" title="{title}">{title}</div>
    <div class="card-meta">
      <span class="pill pill-amber">Predicted {score:.1f} ★</span>
    </div>
  </div>
</div>"""


def _render_grid(df: pd.DataFrame, score_col: str, score_cls: str):
    """
    Render cards in a CSS grid (not st.columns) so it naturally
    reflows to 2 columns on mobile without any Python logic.
    Each card gets its own expander rendered below it in a second grid row.
    """
    # Build all card HTML in one block — CSS grid handles columns
    cards_html = '<div class="card-grid">'
    for _, row in df.iterrows():
        score = float(row.get(score_col, 0) or 0)
        if score_cls == "score-l":
            cards_html += _ml_card_html(row, score)
        else:
            cards_html += _card_html(row, score, score_cls)
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    # Expanders below (Streamlit native, so they work on mobile too)
    with st.expander("📋 View Details for All Results"):
        for _, row in df.iterrows():
            title = str(row.get("title",""))
            plot  = str(row.get("plot","") or row.get("overview","") or "No plot available.")
            director = str(row.get("director","") or "")
            actors   = str(row.get("actors","")   or "")
            score    = float(row.get(score_col, 0) or 0)
            st.markdown(
                f"**{title}** &nbsp; "
                f"<span style='color:var(--muted);font-size:.8rem'>{score:.3f} score</span>",
                unsafe_allow_html=True,
            )
            st.caption(plot[:220] + ("…" if len(plot) > 220 else ""))
            if director: st.caption(f"🎬 {director}")
            if actors:   st.caption(f"🎭 {actors[:80]}")
            st.markdown("---")


# ─────────────────────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <div class="hero-logo">CineMatch AI</div>
  <p class="hero-sub">Netflix-Style Recommendation Engine · Powered by OMDb API & MovieLens 100K</p>
  <div class="hero-badges">
    <span class="hbadge hbadge-t">TF-IDF Content Engine</span>
    <span class="hbadge hbadge-a">SVD Collaborative Filter</span>
    <span class="hbadge hbadge-r">Hybrid Recommender</span>
    <span class="hbadge hbadge-v">OMDb · No Geo-Blocks</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sb-logo">🎬 CineMatch</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-tag">AI RECOMMENDATION ENGINE</div>', unsafe_allow_html=True)

    mode = st.radio(
        "Mode",
        ["🔍 Content-Based", "👥 Collaborative", "⚡ Hybrid"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    n_recs = st.slider("Recommendations", 5, 20, TOP_N)
    st.markdown("---")

    st.markdown("""
    <div class="sb-desc">
      <b class="t">Content-Based</b><br>
      TF-IDF vectors over plot, genres, director & cast.
      Cosine similarity finds the closest films — no user data needed.<br><br>
      <b class="a">Collaborative</b><br>
      SVD matrix factorisation on 100K MovieLens ratings predicts
      what unseen movies a user would enjoy.<br><br>
      <b class="r">Hybrid</b><br>
      Weighted blend of both: <i>α·content + β·collab</i>.
      Tune the slider to explore the full spectrum.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="display:flex;flex-wrap:wrap;gap:.4rem;justify-content:center">
      <div class="stat-chip"><div class="val">{len(omdb_df)}</div><div class="lbl">OMDb Movies</div></div>
      <div class="stat-chip"><div class="val">{len(ratings_df):,}</div><div class="lbl">ML Ratings</div></div>
      <div class="stat-chip"><div class="val">{ratings_df['user_id'].nunique()}</div><div class="lbl">Users</div></div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  MODE 1 — Content-Based Filtering
# ─────────────────────────────────────────────────────────────

if mode == "🔍 Content-Based":
    st.markdown('<div class="sh">Content-Based <span>Filtering</span></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="infobox">
      <b>How it works:</b> Each movie is vectorised with <b>TF-IDF</b> over its
      plot, genres, director, actors and country.
      <b>Cosine similarity</b> ranks every film by feature-space proximity.
    </div>
    """, unsafe_allow_html=True)

    selected = st.selectbox("🎥 Pick a movie:", content_rec.all_titles())

    if selected:
        sel     = omdb_df[omdb_df["title"] == selected].iloc[0]
        poster  = poster_from_row(sel)
        rating  = float(sel.get("imdb_rating", 0) or 0)
        runtime = str(sel.get("runtime",  "") or "")
        rated   = str(sel.get("rated",    "") or "")
        year    = str(sel.get("year",     "") or "")[:4]
        rt      = str(sel.get("rt_rating","") or "")

        st.markdown(f"""
        <div class="movie-hero">
          <img src="{poster}" alt="{selected}" onerror="this.onerror=null;this.src=''"/>
          <div class="movie-hero-info">
            <h2>{sel['title']}</h2>
            <div style="margin-bottom:.6rem">
              {_stars(rating)}
              <span style="color:var(--muted);font-size:.85rem;margin-left:.4rem">
                IMDb {rating}/10
                {"· 🍅 " + rt if rt else ""}
                · {year} · {runtime}
                {"· " + rated if rated and rated!="N/A" else ""}
              </span>
            </div>
            <div style="margin-bottom:.7rem">{_genre_pills(sel.get("genre",""))}</div>
            <p style="color:#94a3b8;font-size:.88rem;line-height:1.6;margin:0 0 .6rem">
              {str(sel.get("plot","") or "")[:350]}
            </p>
            {"<div style='font-size:.82rem;color:var(--muted)'>🎬 <b style=color:var(--text)>Director:</b> " + str(sel.get("director","")) + "</div>" if sel.get("director") else ""}
            {"<div style='font-size:.82rem;color:var(--muted);margin-top:.3rem'>🎭 <b style=color:var(--text)>Cast:</b> " + str(sel.get("actors",""))[:120] + "</div>" if sel.get("actors") else ""}
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        with st.spinner("🔍 Finding similar movies…"):
            recs = content_rec.recommend(selected, n=n_recs)

        if not recs.empty:
            st.markdown(f'<div class="sh">Similar to <span>{selected}</span></div>', unsafe_allow_html=True)
            _render_grid(recs, "similarity_score", "score-c")
        else:
            st.warning("No similar movies found. Try another title.")


# ─────────────────────────────────────────────────────────────
#  MODE 2 — Collaborative Filtering
# ─────────────────────────────────────────────────────────────

elif mode == "👥 Collaborative":
    st.markdown('<div class="sh sh-amber">Collaborative <span>Filtering</span></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="infobox infobox-amber">
      <b>How it works:</b> A <b>943 × 1,682</b> user-item matrix from MovieLens 100K
      is decomposed via <b>Truncated SVD</b>. Latent factors predict ratings a user
      has never given — sorted to surface their best unseen films.
    </div>
    """, unsafe_allow_html=True)

    uid = st.selectbox(
        "👤 Select a User ID (1–943):",
        collab_rec.all_user_ids, index=2,
    )

    if uid:
        history      = collab_rec.user_history(uid, n=6)
        user_ratings = ratings_df[ratings_df["user_id"] == uid]
        avg          = user_ratings["rating"].mean()

        # Stats row — responsive flex
        st.markdown(f"""
        <div style="display:flex;flex-wrap:wrap;gap:.5rem;margin-bottom:1rem">
          <div class="stat-chip"><div class="val">{len(user_ratings)}</div><div class="lbl">Rated</div></div>
          <div class="stat-chip"><div class="val">{avg:.1f}★</div><div class="lbl">Avg</div></div>
        </div>
        """, unsafe_allow_html=True)

        # History card
        st.markdown(f"""
        <div class="user-card">
          <div class="user-card-title">📋 User {uid} — Highest Rated</div>
        """, unsafe_allow_html=True)
        for _, h in history.iterrows():
            r = int(h.get("rating", 0))
            st.markdown(
                f"**{h['title']}** &nbsp;"
                f"<span style='color:var(--amber)'>{'★'*r}{'☆'*(5-r)}</span> "
                f"<span style='color:var(--muted);font-size:.8rem'>{r}/5</span>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        with st.spinner(f"🤖 Predicting top picks for User {uid}…"):
            recs = collab_rec.recommend(uid, n=n_recs)

        if not recs.empty:
            st.markdown(f'<div class="sh sh-amber">Predicted <span>Top Picks</span> for User {uid}</div>', unsafe_allow_html=True)
            _render_grid(recs, "predicted_rating", "score-l")
        else:
            st.warning("Insufficient data for this user. Try another User ID.")


# ─────────────────────────────────────────────────────────────
#  MODE 3 — Hybrid Recommender
# ─────────────────────────────────────────────────────────────

elif mode == "⚡ Hybrid":
    st.markdown('<div class="sh sh-rose">Hybrid <span>Recommender</span></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="infobox infobox-rose">
      <b>How it works:</b> Blends <b>Content</b> (TF-IDF cosine) and
      <b>Collaborative</b> (SVD predicted rating) — both normalised to [0,1].
      <b>hybrid = α·content + β·collab</b>. Adjust below in real time.
    </div>
    """, unsafe_allow_html=True)

    # On mobile these stack vertically via the CSS override above
    ca, cb = st.columns(2)
    with ca:
        seed  = st.selectbox("🎥 Seed Movie:", content_rec.all_titles())
    with cb:
        h_uid = st.selectbox("👤 User ID:", collab_rec.all_user_ids, index=2)

    cw = st.slider(
        "← Content  |  Blend  |  Collaborative →",
        0.0, 1.0, 0.4, 0.05,
        help="0 = full collaborative · 1 = full content-based",
    )
    lw = round(1.0 - cw, 2)

    st.markdown(f"""
    <p style="font-size:.84rem;color:var(--muted);margin-bottom:1rem">
      Blend: <b style="color:var(--teal)">{cw:.0%} Content</b>
      &nbsp;+&nbsp;
      <b style="color:var(--amber)">{lw:.0%} Collaborative</b>
    </p>
    """, unsafe_allow_html=True)

    hybrid_rec.cw = cw
    hybrid_rec.lw = lw

    if seed and h_uid:
        with st.spinner("⚡ Running hybrid engine…"):
            recs = hybrid_rec.recommend(h_uid, seed, n=n_recs)

        if not recs.empty:
            st.markdown(
                f'<div class="sh sh-rose">Hybrid Picks: <span>{seed}</span> × User {h_uid}</div>',
                unsafe_allow_html=True,
            )

            with st.expander("📊 Score Breakdown"):
                show_cols = [c for c in ["title","content_score_norm","collab_score_norm","hybrid_score"] if c in recs.columns]
                disp = recs[show_cols].rename(columns={
                    "content_score_norm": "Content",
                    "collab_score_norm":  "Collab",
                    "hybrid_score":       "Hybrid",
                }).reset_index(drop=True)

                def _color(val):
                    try:
                        v = float(val); r2 = int(220*(1-v)); g2 = int(180*v)
                        return f"color:rgb({r2},{g2},80);font-weight:600"
                    except: return ""

                num_cols = [c for c in ["Content","Collab","Hybrid"] if c in disp.columns]
                st.dataframe(
                    disp.style.applymap(_color, subset=num_cols)
                              .format({c: "{:.3f}" for c in num_cols}),
                    use_container_width=True,
                )

            _render_grid(recs, "hybrid_score", "score-h")
        else:
            st.warning("No hybrid results. Try a different seed movie or user.")


# ─────────────────────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center;margin-top:3rem;padding:1.5rem 0;
            border-top:1px solid rgba(255,255,255,0.06);
            color:#334155;font-size:.75rem;letter-spacing:.08em">
  CINEMATCH AI &nbsp;·&nbsp; OMDb API &nbsp;·&nbsp; MovieLens 100K &nbsp;·&nbsp;
  TF-IDF + SVD &nbsp;·&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)
# 🎬 CineMatch AI — Netflix-Style Movie Recommendation System
### OMDb Edition · No Geo-Blocks · Works in India

A production-grade movie recommendation system powered by the **OMDb API** and **MovieLens 100K**, featuring three recommendation engines with a cinematic dark UI built in Streamlit.

---

## ✨ Features

| Engine | Algorithm | Data |
|--------|-----------|------|
| 🔍 Content-Based | TF-IDF + Cosine Similarity | OMDb metadata (150+ movies) |
| 👥 Collaborative | SVD Matrix Factorisation | MovieLens 100K (100K ratings) |
| ⚡ Hybrid | Weighted Blend (live slider) | Both sources |

**No VPN needed** — OMDb API and all image CDNs work freely in India.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run
streamlit run app.py

# 3. Open → http://localhost:8501
```

**First launch:** fetches ~150 movies from OMDb (~3 min). Everything is cached to `.cache/` as Parquet files — subsequent launches are **instant**.

---

## 📁 Project Structure

```
cinematch/
├── app.py            ← Streamlit UI (all 3 modes)
├── recommenders.py   ← ContentRecommender, CollabRecommender, HybridRecommender
├── data_loader.py    ← OMDb fetcher, MovieLens downloader, poster helpers
├── config.py         ← API key, seed movie list, hyperparameters
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration (`config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `OMDB_SEED_MOVIES` | 150+ titles | Movies to build the content catalogue |
| `SVD_N_FACTORS` | 100 | SVD latent factors |
| `HYBRID_CONTENT_WEIGHT` | 0.4 | Default content blend weight |
| `HYBRID_COLLAB_WEIGHT` | 0.6 | Default collab blend weight |
| `TOP_N` | 10 | Default recommendations shown |

### Adding more movies
Just append titles to `OMDB_SEED_MOVIES` in `config.py`, then delete `.cache/omdb_movies.parquet` to force a refresh.

---

## 🧠 How Each Engine Works

### 1. Content-Based (TF-IDF + Cosine Similarity)
Each movie's **"soup"** = `plot×1 + genre×2 + director×2 + actors×1 + country×1`  
TF-IDF (12K features, bigrams, sublinear TF) → cosine similarity matrix → top-N neighbours.

### 2. Collaborative (SVD Matrix Factorisation)
943 × 1682 rating matrix from MovieLens 100K → user means subtracted → TruncatedSVD (k=100) → reconstruct full predicted matrix → rank unseen movies by predicted rating for target user.

### 3. Hybrid (Weighted Blend)
1. Content engine returns top-60 candidate movies  
2. SVD predicts rating for each candidate for target user  
3. Both scores normalised to [0,1] via min-max  
4. `hybrid = α × content + β × collab` (α+β=1, configurable via slider)

---

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| OMDb returns no results | Check your API key in `config.py` |
| Slow first load | Normal — fetches ~150 movies. Cached after first run. |
| Missing posters | OMDb occasionally has no poster for older films; SVG card shown instead |
| Want more movies | Add titles to `OMDB_SEED_MOVIES` and delete the parquet cache |

---

## 📜 License

MIT — free to use, modify, and distribute.

---

*Built with Python · Streamlit · scikit-learn · OMDb API · MovieLens 100K*

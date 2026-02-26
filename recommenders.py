# ============================================================
#  CineMatch AI — Recommendation Engines
#
#  1. ContentRecommender  — TF-IDF + Cosine Similarity
#  2. CollabRecommender   — SVD Matrix Factorisation (sklearn)
#  3. HybridRecommender   — Weighted blend of 1 + 2
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder

from config import SVD_N_FACTORS, SVD_N_EPOCHS, TOP_N
from config import HYBRID_CONTENT_WEIGHT, HYBRID_COLLAB_WEIGHT


# ═══════════════════════════════════════════════════════════
#  Utility
# ═══════════════════════════════════════════════════════════

def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


# ═══════════════════════════════════════════════════════════
#  1. Content-Based — TF-IDF + Cosine Similarity
# ═══════════════════════════════════════════════════════════

class ContentRecommender:
    """
    Vectorises each movie's 'soup' (plot + genre + director +
    actors + country) with TF-IDF, then uses cosine similarity
    to find the nearest neighbours.
    """

    def __init__(self, movies_df: pd.DataFrame):
        self.df = movies_df.reset_index(drop=True)
        self._fit()

    def _fit(self):
        tfidf = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=12_000,
            sublinear_tf=True,
        )
        mat = tfidf.fit_transform(self.df["soup"].fillna(""))
        self.sim = linear_kernel(mat, mat)          # full similarity matrix
        self._title_idx = pd.Series(
            self.df.index,
            index=self.df["title"].str.lower()
        )

    # ── Public ────────────────────────────────────────────
    def recommend(self, title: str, n: int = TOP_N) -> pd.DataFrame:
        """Top-n similar movies for a given title."""
        idx = self._resolve(title)
        if idx is None:
            return pd.DataFrame()
        scores = list(enumerate(self.sim[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [(i, s) for i, s in scores if i != idx][:n]
        result = self.df.iloc[[i for i, _ in scores]].copy()
        result["similarity_score"] = [s for _, s in scores]
        return result.reset_index(drop=True)

    def score(self, title_a: str, title_b: str) -> float:
        ia, ib = self._resolve(title_a), self._resolve(title_b)
        if ia is None or ib is None:
            return 0.0
        return float(self.sim[ia, ib])

    def all_titles(self) -> list[str]:
        return self.df["title"].tolist()

    def _resolve(self, title: str):
        key = title.lower()
        if key in self._title_idx:
            return self._title_idx[key]
        # prefix fallback
        hits = [t for t in self._title_idx.index if t.startswith(key[:8])]
        return self._title_idx[hits[0]] if hits else None


# ═══════════════════════════════════════════════════════════
#  2. Collaborative — SVD Matrix Factorisation
# ═══════════════════════════════════════════════════════════

class CollabRecommender:
    """
    Builds a user × item matrix from MovieLens 100K ratings,
    decomposes it with TruncatedSVD, then predicts unseen ratings.
    """

    def __init__(self, ratings_df: pd.DataFrame, ml_movies_df: pd.DataFrame):
        self.ratings   = ratings_df
        self.ml_movies = ml_movies_df
        self._fit()

    def _fit(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

        r = self.ratings.copy()
        r["u"] = self.user_enc.fit_transform(r["user_id"])
        r["v"] = self.item_enc.fit_transform(r["item_id"])

        n_u = r["u"].nunique()
        n_v = r["v"].nunique()

        # Build rating matrix
        R = np.zeros((n_u, n_v), dtype=np.float32)
        for _, row in r.iterrows():
            R[int(row["u"]), int(row["v"])] = row["rating"]

        # Centre by user mean
        self._umeans = np.where(
            (R != 0).sum(1) > 0,
            R.sum(1) / (R != 0).sum(1).clip(min=1),
            0
        )
        Rc = R.copy()
        nz = R != 0
        Rc[nz] -= self._umeans[np.where(nz)[0]]

        # SVD
        k   = min(SVD_N_FACTORS, min(n_u, n_v) - 1)
        svd = TruncatedSVD(n_components=k, n_iter=SVD_N_EPOCHS, random_state=42)
        U   = svd.fit_transform(Rc)
        Vt  = svd.components_

        self._Rpred = U @ Vt + self._umeans[:, None]

        self._iidx  = {iid: i for i, iid in enumerate(self.item_enc.classes_)}
        self._irev  = dict(enumerate(self.item_enc.classes_))
        self.all_user_ids = sorted(self.ratings["user_id"].unique().tolist())

    # ── Public ────────────────────────────────────────────
    def recommend(self, user_id: int, n: int = TOP_N) -> pd.DataFrame:
        """Top-n unseen movies for user_id."""
        if user_id not in self.user_enc.classes_:
            return pd.DataFrame()
        u = int(self.user_enc.transform([user_id])[0])
        seen = set(self.ratings[self.ratings["user_id"] == user_id]["item_id"])

        preds = [
            (self._irev[i], float(s))
            for i, s in enumerate(self._Rpred[u])
            if self._irev[i] not in seen
        ]
        preds.sort(key=lambda x: x[1], reverse=True)
        preds = preds[:n]

        item_ids = [p[0] for p in preds]
        scores   = {p[0]: p[1] for p in preds}

        result = self.ml_movies[self.ml_movies["item_id"].isin(item_ids)].copy()
        result["predicted_rating"] = result["item_id"].map(scores)
        return result.sort_values("predicted_rating", ascending=False).reset_index(drop=True)

    def predict(self, user_id: int, item_id: int) -> float:
        if user_id not in self.user_enc.classes_: return 0.0
        if item_id not in self._iidx:             return 0.0
        u = int(self.user_enc.transform([user_id])[0])
        return float(self._Rpred[u, self._iidx[item_id]])

    def user_history(self, user_id: int, n: int = 5) -> pd.DataFrame:
        h = self.ratings[self.ratings["user_id"] == user_id].copy()
        h = h.merge(self.ml_movies, on="item_id", how="left")
        return h.sort_values("rating", ascending=False).head(n)


# ═══════════════════════════════════════════════════════════
#  3. Hybrid — Weighted Blend
# ═══════════════════════════════════════════════════════════

class HybridRecommender:
    """
    score = α × content_cosine  +  β × norm(SVD_predicted_rating)
    Both scores normalised to [0,1] before blending.
    """

    def __init__(
        self,
        content: ContentRecommender,
        collab:  CollabRecommender,
        cw: float = HYBRID_CONTENT_WEIGHT,
        lw: float = HYBRID_COLLAB_WEIGHT,
    ):
        self.cr  = content
        self.col = collab
        self.cw  = cw
        self.lw  = lw

    def recommend(
        self,
        user_id:    int,
        seed_movie: str,
        n:          int = TOP_N,
        pool:       int = 60,
    ) -> pd.DataFrame:
        # Step 1 — content candidates
        cands = self.cr.recommend(seed_movie, n=pool)
        if cands.empty:
            return pd.DataFrame()

        # Step 2 — collab scores per candidate
        collab_raw = []
        for _, row in cands.iterrows():
            iid = self._match_ml(row["title"])
            collab_raw.append(
                self.col.predict(user_id, iid) if iid else 0.0
            )

        cands = cands.copy()
        cands["collab_raw"] = collab_raw

        # Step 3 — normalise both to [0,1]
        cs = _minmax(cands["similarity_score"].values.astype(float))
        ls = _minmax(cands["collab_raw"].values.astype(float))

        cands["content_score_norm"] = cs
        cands["collab_score_norm"]  = ls
        cands["hybrid_score"] = self.cw * cs + self.lw * ls

        return (
            cands.sort_values("hybrid_score", ascending=False)
                 .head(n)
                 .reset_index(drop=True)
        )

    def _match_ml(self, omdb_title: str):
        """Fuzzy-match OMDb title → MovieLens item_id."""
        ml = self.col.ml_movies["title"].str.lower()
        key = omdb_title.lower().split("(")[0].strip()
        hit = self.col.ml_movies[ml.str.contains(key[:10], regex=False, na=False)]
        return int(hit.iloc[0]["item_id"]) if not hit.empty else None

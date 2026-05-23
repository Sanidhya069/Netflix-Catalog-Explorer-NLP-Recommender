import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# New CSV headers (netflix_titles.csv)
NEW_SCHEMA = {
    "title": "Title",
    "description": "Summary",
    "genre": "Genre",
    "tags": "Tags",
    "type": "Series or Movie",
    "release_date": "Netflix Release Date",
}
LEGACY_SCHEMA = {"title", "description", "listed_in"}

def _combine_text(row: pd.Series) -> str:
    parts = [
        str(row.get("description") or "").strip(),
        str(row.get("listed_in") or "").strip(),
    ]
    text = " ".join(p for p in parts if p)
    return text if text else "no description"

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map new or legacy CSV columns to canonical names used by the app."""
    df = df.copy()
    cols = set(df.columns)

    if NEW_SCHEMA["title"] in cols:
        df["title"] = df[NEW_SCHEMA["title"]].fillna("").astype(str)
        df["description"] = (
            df[NEW_SCHEMA["description"]].fillna("").astype(str)
            if NEW_SCHEMA["description"] in cols
            else ""
        )
        genre = (
            df[NEW_SCHEMA["genre"]].fillna("").astype(str)
            if NEW_SCHEMA["genre"] in cols
            else ""
        )
        tags = (
            df[NEW_SCHEMA["tags"]].fillna("").astype(str)
            if NEW_SCHEMA["tags"] in cols
            else ""
        )
        df["listed_in"] = (genre + " " + tags).str.strip()
        if NEW_SCHEMA["type"] in cols:
            df["type"] = df[NEW_SCHEMA["type"]]
        if NEW_SCHEMA["release_date"] in cols:
            df["release_year"] = pd.to_datetime(
                df[NEW_SCHEMA["release_date"]], errors="coerce"
            ).dt.year
        elif "Release Date" in cols:
            df["release_year"] = pd.to_datetime(
                df["Release Date"], errors="coerce"
            ).dt.year
        return df

    if LEGACY_SCHEMA.issubset(cols):
        df["title"] = df["title"].fillna("").astype(str)
        df["description"] = df["description"].fillna("").astype(str)
        df["listed_in"] = df["listed_in"].fillna("").astype(str)
        return df

    raise ValueError(
        "Unrecognized CSV schema. Expected new columns "
        f"{list(NEW_SCHEMA.values())} or legacy columns {sorted(LEGACY_SCHEMA)}."
    )

@st.cache_data(show_spinner="Loading dataset and building TF-IDF index…")
def load_data_and_index(csv_path: str) -> tuple[pd.DataFrame, TfidfVectorizer, sparse.spmatrix]:
    df = _normalize_columns(pd.read_csv(csv_path))
    df["_search_text"] = df.apply(_combine_text, axis=1)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=12_000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(df["_search_text"])
    return df, vectorizer, tfidf

def recommend(
    vectorizer: TfidfVectorizer,
    tfidf_matrix: sparse.spmatrix,
    query: str,
    top_n: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    q = (query or "").strip()
    if not q:
        return np.array([], dtype=int), np.array([], dtype=float)
    q_vec = vectorizer.transform([q])
    sims = cosine_similarity(q_vec, tfidf_matrix).ravel()
    n = min(top_n, sims.size)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    idx = np.argpartition(-sims, n - 1)[:n]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]
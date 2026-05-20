"""
Netflix TF-IDF recommender (home page).
Run: streamlit run app.py
"""

from __future__ import annotations

import html
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = Path(__file__).resolve().parent / "netflix_titles.csv"


def _combine_text(row: pd.Series) -> str:
    parts = [
        str(row.get("description") or "").strip(),
        str(row.get("listed_in") or "").strip(),
    ]
    text = " ".join(p for p in parts if p)
    return text if text else "no description"


@st.cache_data(show_spinner="Loading dataset and building TF-IDF index…")
def load_data_and_index(csv_path: str) -> tuple[pd.DataFrame, TfidfVectorizer, sparse.spmatrix]:
    df = pd.read_csv(csv_path)
    needed = {"title", "description", "listed_in"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    df = df.copy()
    df["description"] = df["description"].fillna("")
    df["listed_in"] = df["listed_in"].fillna("")
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


@st.cache_data(show_spinner=False)
def fetch_tmdb_data(title: str, media_type: str, release_year: str, api_key: str) -> tuple[str | None, float | None]:
    """
    Fetches the poster path and average rating from TMDB API.
    Returns (poster_url, vote_average).
    """
    if not api_key:
        return None, None

    is_movie = str(media_type).lower() == "movie"
    endpoint = "movie" if is_movie else "tv"
    url = f"https://api.themoviedb.org/3/search/{endpoint}"
    
    headers = {
        "accept": "application/json"
    }
    
    params = {
        "api_key": api_key,
        "query": title,
    }
    
    # Try searching with release year first for better accuracy
    if release_year:
        if is_movie:
            params["year"] = str(release_year)
        else:
            params["first_air_date_year"] = str(release_year)
            
    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if results:
                best_match = results[0]
                poster_path = best_match.get("poster_path")
                vote_average = best_match.get("vote_average")
                poster_url = f"https://image.tmdb.org/t/p/w342{poster_path}" if poster_path else None
                return poster_url, vote_average
                
        # If we got no results with year, try searching without the year filter
        if "year" in params or "first_air_date_year" in params:
            params.pop("year", None)
            params.pop("first_air_date_year", None)
            response = requests.get(url, params=params, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                if results:
                    best_match = results[0]
                    poster_path = best_match.get("poster_path")
                    vote_average = best_match.get("vote_average")
                    poster_url = f"https://image.tmdb.org/t/p/w342{poster_path}" if poster_path else None
                    return poster_url, vote_average
                    
    except Exception:
        # Gracefully swallow connection/timeout issues and return None
        pass
        
    return None, None


st.set_page_config(
    page_title="Netflix Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("Configuration")
st.sidebar.markdown(
    "Enhance recommendations with live movie posters and ratings from TMDB."
)
tmdb_api_key = st.sidebar.text_input(
    "TMDB API Key",
    type="password",
    help="Enter your TMDB API Key. You can get one for free at themoviedb.org",
    value=os.getenv("TMDB_API_KEY", "")
)

if not tmdb_api_key:
    st.sidebar.info(
        "💡 Enter your TMDB API Key to see movie posters and official ratings."
    )


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; max-width: 1100px; }
    .rec-card {
        border: 1px solid rgba(229, 9, 20, 0.35);
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.65rem;
        background: linear-gradient(145deg, rgba(20,20,22,0.95), rgba(30,30,34,0.9));
    }
    .rec-title { font-weight: 600; font-size: 1.05rem; margin-bottom: 0.25rem; }
    .rec-meta { color: #aaa; font-size: 0.82rem; margin-bottom: 0.4rem; }
    .rec-desc { color: #ddd; font-size: 0.88rem; line-height: 1.45; }
    .sim-badge {
        display: inline-block;
        background: #E50914;
        color: white;
        padding: 0.12rem 0.45rem;
        border-radius: 6px;
        font-size: 0.75rem;
        margin-left: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not DATA_PATH.is_file():
    st.error(f"Dataset not found at `{DATA_PATH}`. Place `netflix_titles.csv` next to `app.py`.")
    st.stop()

try:
    df, vectorizer, tfidf_matrix = load_data_and_index(str(DATA_PATH))
except Exception as e:
    st.exception(e)
    st.stop()

st.title("Netflix catalog explorer")
st.caption(
    "Content-based recommendations using TF-IDF over **description** and **listed_in** (genres). "
    "Open **Dashboard** in the sidebar for catalog metrics and charts."
)

st.subheader("TF-IDF recommender")
st.markdown(
    "Describe a mood, theme, or genres (e.g. *dark comedy heist europe*). "
    "Matches are ranked by cosine similarity to catalog text."
)

with st.form("search_form", clear_on_submit=False):
    query = st.text_input(
        "Search",
        placeholder="Try: psychological thriller serial killer investigation",
        label_visibility="collapsed",
    )
    top_n = st.slider("Number of results", min_value=5, max_value=30, value=12, step=1)
    submitted = st.form_submit_button("Recommend", type="primary")

if submitted:
    if not query.strip():
        st.session_state.pop("rec_pack", None)
        st.info("Enter a search query to see recommendations.")
    else:
        idx, scores = recommend(vectorizer, tfidf_matrix, query, top_n=top_n)
        if len(idx) == 0:
            st.session_state.pop("rec_pack", None)
            st.warning("No results.")
        else:
            st.session_state["rec_pack"] = (idx.tolist(), scores.tolist(), query)

rec_pack = st.session_state.get("rec_pack")
if rec_pack is not None:
    idx_list, score_list, _q = rec_pack
    st.caption(f"Showing {len(idx_list)} matches (cosine similarity × 100).")
    
    cols_per_row = 4
    for i in range(0, len(idx_list), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i + j
            if idx < len(idx_list):
                row_idx = idx_list[idx]
                row = df.iloc[int(row_idx)]
                
                title = str(row.get("title", "Untitled"))
                typ = str(row.get("type", ""))
                year = str(row.get("release_year", ""))
                listed = str(row.get("listed_in", "") or "")
                raw_desc = str(row.get("description") or "")
                desc = raw_desc[:200]
                if len(raw_desc) > 200:
                    desc += "…"
                
                score_pct = float(score_list[idx]) * 100
                
                # Fetch live movie info (poster & rating) from TMDB
                poster_url, rating = fetch_tmdb_data(title, typ, year, tmdb_api_key)
                
                with cols[j]:
                    # Display poster or a beautiful colored fallback placeholder
                    if poster_url:
                        st.image(poster_url, use_container_width=True)
                    else:
                        st.markdown(
                            f"""
                            <div style="
                                background: linear-gradient(135deg, #1e1e24, #121214);
                                border: 1.5px dashed rgba(229, 9, 20, 0.4);
                                border-radius: 8px;
                                aspect-ratio: 2/3;
                                display: flex;
                                flex-direction: column;
                                justify-content: center;
                                align-items: center;
                                text-align: center;
                                padding: 1.25rem;
                                color: #fff;
                                margin-bottom: 8px;
                                box-shadow: inset 0 0 15px rgba(229, 9, 20, 0.1);
                            ">
                                <div style="font-size: 2.2rem; margin-bottom: 8px;">🎬</div>
                                <div style="font-weight: 700; font-size: 0.95rem; line-height: 1.3; overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;">
                                    {html.escape(title)}
                                </div>
                                <div style="color: #888; font-size: 0.75rem; margin-top: 6px;">
                                    {html.escape(typ)} · {html.escape(year)}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Movie Details & TMDB Rating
                    rating_str = f"⭐ {rating:.1f}" if rating is not None else "⭐ N/A"
                    st.markdown(
                        f"""
                        <div style="padding: 2px 4px 12px 4px;">
                            <div style="font-weight: 700; font-size: 1.05rem; line-height: 1.3; color: #fff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 2px;" title="{html.escape(title)}">
                                {html.escape(title)}
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                <span style="background-color: rgba(229, 9, 20, 0.2); color: #E50914; padding: 2px 6px; border-radius: 4px; font-weight: 600; font-size: 0.75rem;">
                                    {score_pct:.1f}% Match
                                </span>
                                <span style="color: #ffb61e; font-weight: 600; font-size: 0.85rem;">
                                    {rating_str}
                                </span>
                            </div>
                            <div style="color: #aaa; font-size: 0.78rem; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 6px;" title="{html.escape(listed)}">
                                <b>Genres:</b> {html.escape(listed)}
                            </div>
                            <p style="color: #ddd; font-size: 0.78rem; line-height: 1.4; margin: 0; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;" title="{html.escape(raw_desc)}">
                                {html.escape(desc)}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


"""
Netflix catalog explorer with TF-IDF content recommendations (description + listed_in).
Run: streamlit run app.py
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
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
    # dense is heavy for 8k rows but fine; keep sparse and use linear_kernel is faster
    # cosine_similarity accepts sparse; return sparse csr for memory
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


def main() -> None:
    st.set_page_config(
        page_title="Netflix Catalog",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; max-width: 1100px; }
        div[data-testid="stMetricValue"] { font-size: 1.75rem; }
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
        "Content-based recommendations using TF-IDF over **description** and **listed_in** (genres)."
    )

    type_counts = df["type"].value_counts() if "type" in df.columns else pd.Series(dtype=int)
    n_movies = int(type_counts.get("Movie", 0))
    n_shows = int(type_counts.get("TV Show", 0))
    n_total = len(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Titles", f"{n_total:,}")
    c2.metric("Movies", f"{n_movies:,}")
    c3.metric("TV shows", f"{n_shows:,}")

    st.subheader("Overview")
    oc1, oc2 = st.columns((1, 1), gap="large")

    with oc1:
        st.markdown("**Movies vs TV shows**")
        if not type_counts.empty:
            chart_df = type_counts.rename_axis("type").reset_index(name="count")
            st.bar_chart(chart_df.set_index("type"), height=280)
        else:
            st.info("No `type` column for this chart.")

    with oc2:
        st.markdown("**Top countries** (as listed in the dataset)")
        if "country" in df.columns:
            top_c = (
                df["country"]
                .fillna("Unknown")
                .value_counts()
                .head(10)
                .rename_axis("country")
                .reset_index(name="titles")
            )
            st.bar_chart(top_c.set_index("country"), height=280)
        else:
            st.info("No `country` column.")

    st.divider()
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
        for i, row_idx in enumerate(idx_list):
            row = df.iloc[int(row_idx)]
            title = html.escape(str(row.get("title", "Untitled")))
            typ = html.escape(str(row.get("type", "")))
            year = html.escape(str(row.get("release_year", "")))
            listed = html.escape(str(row.get("listed_in", "") or ""))
            raw_desc = str(row.get("description") or "")
            desc = html.escape(raw_desc[:320])
            if len(raw_desc) > 320:
                desc += "…"
            score_pct = float(score_list[i]) * 100
            st.markdown(
                f"""
                <div class="rec-card">
                    <div class="rec-title">{title}
                        <span class="sim-badge">{score_pct:.1f}% match</span>
                    </div>
                    <div class="rec-meta">{typ} · {year}</div>
                    <div class="rec-meta"><b>Genres:</b> {listed}</div>
                    <div class="rec-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("Raw data preview"):
        show_cols = [
            c
            for c in [
                "title",
                "type",
                "release_year",
                "rating",
                "listed_in",
                "description",
            ]
            if c in df.columns
        ]
        st.dataframe(df[show_cols].head(50), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

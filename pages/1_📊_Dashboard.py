"""Netflix catalog dashboard — metrics and overview charts."""

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent.parent / "netflix_titles.csv"


@st.cache_data(show_spinner="Loading dataset…")
def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; max-width: 1100px; }
    div[data-testid="stMetricValue"] { font-size: 1.75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Catalog dashboard")
st.caption("High-level metrics and distributions from the Netflix titles dataset.")

if not DATA_PATH.is_file():
    st.error(f"Dataset not found at `{DATA_PATH}`. Place `netflix_titles.csv` in the project root.")
    st.stop()

try:
    df = load_data(str(DATA_PATH))
except Exception as e:
    st.exception(e)
    st.stop()

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

with st.expander("Raw data preview"):
    show_cols = [
        c
        for c in [
            "title",
            "type",
            "release_year",
            "rating",
            "country",
            "listed_in",
            "description",
        ]
        if c in df.columns
    ]
    st.dataframe(df[show_cols].head(50), use_container_width=True, hide_index=True)

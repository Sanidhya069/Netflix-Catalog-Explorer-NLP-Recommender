# 🎬 Netflix Catalog Explorer & NLP Recommender

An interactive web application built with Python and Streamlit that allows users to explore the Netflix catalog and discover new content through an AI-powered recommendation engine. 

Instead of relying on basic keyword searches, this tool uses Natural Language Processing (NLP) to understand the mood, theme, and genres you are looking for, matching you with the perfect movie or TV show.

## ✨ Features
* **Smart Recommendation Engine:** Uses Term Frequency-Inverse Document Frequency (TF-IDF) and cosine similarity to find highly relevant content based on user descriptions (e.g., "gritty crime drama in Europe").
* **Interactive UI:** A custom, dark-themed interface built entirely in Python using Streamlit, featuring real-time data filtering.
* **Dynamic Data Dashboard:** Visualizes catalog metrics, including Movie vs. TV Show ratios and top producing countries, using responsive bar charts.
* **Persistent Search:** Utilizes session states to ensure search results remain visible while navigating and interacting with other dashboard elements.

## 🛠️ Tech Stack
* **Language:** Python
* **Frontend/UI:** Streamlit
* **Machine Learning / NLP:** Scikit-learn (TF-IDF Vectorization, Cosine Similarity)
* **Data Manipulation:** Pandas, NumPy

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Sanidhya069/netflix-data-analysis.git](https://github.com/Sanidhya069/netflix-data-analysis.git)
   cd netflix-data-analysis
2. **Install the required dependencies:**
    pip install -r requirements.txt
3. **Launch the Streamlit app:**
     streamlit run app.py
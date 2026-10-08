import os
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Smart Content Recommendation",
    page_icon="🎬",
    layout="centered"
)


# =========================================================
# SIMPLE DESIGN
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #0f1020;
}

h1, h2, h3, h4, p, label {
    color: white !important;
}

.title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
    margin-top: 20px;
}

.subtitle {
    text-align: center;
    color: #bbbbcc;
    margin-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title">🎬 Smart Content Recommendation</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Content-Based Filtering using TF-IDF + Cosine Similarity'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# FIND CSV
# =========================================================

current_folder = os.path.dirname(
    os.path.abspath(__file__)
)

csv_path = os.path.join(
    current_folder,
    "movies.csv"
)


# =========================================================
# LOAD CSV
# =========================================================

if not os.path.exists(csv_path):

    st.error("❌ movies.csv was not found.")

    st.write("Your folder must contain:")

    st.code("""
C:\\social media analysis\\
│
├── app.py
└── movies.csv
""")

    st.stop()


try:

    df = pd.read_csv(
        csv_path,
        encoding="utf-8-sig"
    )

except Exception as e:

    st.error("❌ Error reading movies.csv")

    st.code(str(e))

    st.stop()


# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.lower()
)


# =========================================================
# RENAME COLUMNS IF NEEDED
# =========================================================

rename_columns = {}

for col in df.columns:

    clean_col = (
        col
        .replace(" ", "")
        .replace("_", "")
        .lower()
    )

    if clean_col in ["title", "movie", "moviename", "movietitle"]:
        rename_columns[col] = "title"

    elif clean_col in ["genre", "genres", "category"]:
        rename_columns[col] = "genre"

    elif clean_col in ["rating", "ratings", "score", "stars"]:
        rename_columns[col] = "rating"

    elif clean_col in [
        "description",
        "overview",
        "plot",
        "summary"
    ]:
        rename_columns[col] = "description"


df = df.rename(
    columns=rename_columns
)


# =========================================================
# CHECK COLUMNS
# =========================================================

required_columns = [
    "title",
    "genre",
    "rating",
    "description"
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    st.error(
        "❌ Required column missing: "
        + ", ".join(missing_columns)
    )

    st.write("Columns found in your CSV:")

    st.code(
        ", ".join(df.columns.tolist())
    )

    st.write("")

    st.warning(
        "Your first line in movies.csv should be:"
    )

    st.code(
        "title,genre,rating,description"
    )

    st.stop()


# =========================================================
# CLEAN DATA
# =========================================================

df["title"] = (
    df["title"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["genre"] = (
    df["genre"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["description"] = (
    df["description"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# Convert rating into number

df["rating"] = pd.to_numeric(
    df["rating"],
    errors="coerce"
)


# Remove rows without movie title

df = df[
    df["title"] != ""
].copy()


# Remove duplicate movie names

df = df.drop_duplicates(
    subset="title"
)


# Reset index

df = df.reset_index(
    drop=True
)


# =========================================================
# CREATE CONTENT
# =========================================================

# Genre is repeated 3 times
# so genre gets more importance

df["content"] = (
    df["genre"] + " "
) * 3 + df["description"]


# =========================================================
# TF-IDF
# =========================================================

vectorizer = TfidfVectorizer(
    stop_words="english"
)


tfidf_matrix = vectorizer.fit_transform(
    df["content"]
)


# =========================================================
# COSINE SIMILARITY
# =========================================================

similarity_matrix = cosine_similarity(
    tfidf_matrix
)


# =========================================================
# RECOMMENDATION FUNCTION
# =========================================================

def recommend_movies(
    movie_name,
    number_of_movies
):

    # Find selected movie

    selected_rows = df[
        df["title"] == movie_name
    ]


    if selected_rows.empty:

        return pd.DataFrame()


    # Get movie index

    movie_index = selected_rows.index[0]


    # Get similarity scores

    scores = list(
        enumerate(
            similarity_matrix[movie_index]
        )
    )


    # Sort highest similarity first

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )


    # Remove selected movie itself

    scores = [
        item
        for item in scores
        if item[0] != movie_index
    ]


    # Select top movies

    scores = scores[
        :number_of_movies
    ]


    # Movie indexes

    indexes = [
        item[0]
        for item in scores
    ]


    # Similarity percentages

    matches = [
        round(item[1] * 100, 1)
        for item in scores
    ]


    # Create result

    result = df.iloc[indexes].copy()


    result["match"] = matches


    return result.reset_index(
        drop=True
    )


# =========================================================
# MOVIE SELECTION
# =========================================================

st.subheader(
    "🎥 Select a movie you like"
)


movie_list = sorted(
    df["title"].tolist()
)


selected_movie = st.selectbox(
    "Choose a movie:",
    movie_list
)


# =========================================================
# NUMBER OF RECOMMENDATIONS
# =========================================================

number_of_movies = st.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5
)


# =========================================================
# BUTTON
# =========================================================

if st.button(
    "🎯 Find Similar Movies",
    use_container_width=True
):

    # -----------------------------------------------------
    # SELECTED MOVIE
    # -----------------------------------------------------

    selected_movie_data = df[
        df["title"] == selected_movie
    ].iloc[0]


    st.success(
        f"🎯 Because you liked: {selected_movie}"
    )


    # -----------------------------------------------------
    # SELECTED MOVIE DETAILS
    # -----------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        rating = selected_movie_data["rating"]

        if pd.isna(rating):
            st.metric(
                "⭐ Rating",
                "N/A"
            )
        else:
            st.metric(
                "⭐ Rating",
                f"{rating:.1f}"
            )


    with col2:

        st.metric(
            "🎭 Genre",
            selected_movie_data["genre"]
        )


    st.write("")


    # -----------------------------------------------------
    # RECOMMEND MOVIES
    # -----------------------------------------------------

    results = recommend_movies(
        selected_movie,
        number_of_movies
    )


    if results.empty:

        st.warning(
            "No similar movies found."
        )

    else:

        st.subheader(
            "🎬 Recommended for you"
        )


        # -------------------------------------------------
        # DISPLAY RESULTS
        # -------------------------------------------------

        for _, movie in results.iterrows():

            with st.container(border=True):

                st.markdown(
                    f"### 🎬 {movie['title']}"
                )


                col1, col2, col3 = st.columns(3)


                with col1:

                    movie_rating = movie["rating"]

                    if pd.isna(movie_rating):

                        st.write(
                            "⭐ Rating: N/A"
                        )

                    else:

                        st.write(
                            f"⭐ Rating: "
                            f"{movie_rating:.1f}"
                        )


                with col2:

                    st.write(
                        f"🎭 Genre: "
                        f"{movie['genre']}"
                    )


                with col3:

                    st.write(
                        f"🎯 Match: "
                        f"{movie['match']}%"
                    )


                st.write("")


                st.write(
                    movie["description"]
                )


                # Similarity progress bar

                st.progress(
                    min(
                        movie["match"] / 100,
                        1.0
                    )
                )


# =========================================================
# HOW IT WORKS
# =========================================================

st.write("")


with st.expander(
    "ℹ️ How this recommendation system works"
):

    st.write(
        """
        **1. Select a movie**

        **2. The system takes its genre and description**

        **3. TF-IDF converts the text into numerical features**

        **4. Cosine Similarity compares the movies**

        **5. The most similar movies are recommended**
        """
    )


    st.write(
        "**Technologies:** "
        "Python | Pandas | Streamlit | Scikit-learn"
    )
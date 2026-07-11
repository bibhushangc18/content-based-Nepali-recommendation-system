import pickle
import streamlit as st

st.set_page_config(page_title="Nepali Movie Recommender", page_icon="🎬", layout="centered")

@st.cache_resource
def load_data():
    with open("movie_embeddings.pkl", "rb") as f:
        saved = pickle.load(f)
    return saved["df"], saved["similarity_matrix"]

def recommend_by_index(df, similarity_matrix, movie_idx, top_n=5):
    sim_scores = list(enumerate(similarity_matrix[movie_idx]))
    sim_scores = [s for s in sim_scores if s[0] != movie_idx]
    sim_scores.sort(key=lambda x: x[1], reverse=True)
    top = sim_scores[:top_n]
    result = df.loc[[i for i, _ in top], ["title", "genres"]].copy()
    result["similarity_score"] = [round(score, 4) for _, score in top]
    return result.reset_index().rename(columns={"index": "movie_index"})

st.title("🎬 Nepali Movie Recommender")

df, similarity_matrix = load_data()
st.write(f"Catalog: **{len(df)}** movies")

query = st.text_input("Search for a movie you like", placeholder="e.g. Kabaddi")

if query:
    matches = df[df["title"].str.lower().str.contains(query.lower().strip(), na=False)]
    if matches.empty:
        st.warning(f"No movies found matching '{query}'.")
    else:
        options = {
            f"{row.title} ({', '.join(row.genres) if row.genres else 'Unknown'}) [#{idx}]": idx
            for idx, row in matches.iterrows()
        }
        choice = st.selectbox(f"Found {len(matches)} match(es):", options.keys())
        idx = options[choice]
        selected = df.loc[idx]

        st.subheader(f"Because you liked: {selected.title}")
        st.write(f"**Genres:** {', '.join(selected.genres) if selected.genres else 'Unknown'}")

        st.divider()
        st.subheader("Top 5 similar movies")
        recs = recommend_by_index(df, similarity_matrix, idx, top_n=5)
        recs["genres"] = recs["genres"].apply(lambda g: ", ".join(g) if g else "Unknown")
        for _, row in recs.iterrows():
            st.markdown(f"**{row.title}** — {row.genres}")
            st.caption(f"Similarity: {row.similarity_score:.2f}")

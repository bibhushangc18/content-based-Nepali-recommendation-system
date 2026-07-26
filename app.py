import re
import pickle
import streamlit as st

st.set_page_config(page_title="CineNepal", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")

# ---------- Global CSS ----------
st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .stApp {
        background-color: #0b0b0b;
        color: #ffffff;
    }
    h1, h2, h3, .stMarkdown, label, p {
        color: #ffffff !important;
    }

    /* ---- Landing hero ---- */
    .hero-wrap {
        background: linear-gradient(180deg, rgba(11,11,11,0.2) 0%, rgba(11,11,11,0.9) 85%),
                    linear-gradient(120deg, #3a0d0d, #1a0330, #0d1b3a, #1a0330, #3a0d0d);
        background-size: 100% 100%, 400% 400%;
        animation: gradientShift 12s ease infinite;
        border-radius: 12px;
        padding: 80px 40px;
        text-align: center;
        margin-bottom: 24px;
    }
    @keyframes gradientShift {
        0% {background-position: 0% 0%, 0% 50%;}
        50% {background-position: 0% 0%, 100% 50%;}
        100% {background-position: 0% 0%, 0% 50%;}
    }
    .brand-logo {
        font-size: 42px;
        font-weight: 900;
        color: #E50914;
        letter-spacing: 1px;
        margin-bottom: 40px;
    }
    .hero-headline {
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 16px;
    }
    .hero-sub {
        font-size: 18px;
        color: #d2d2d2;
        margin-bottom: 32px;
    }

    /* ---- Buttons ---- */
    div[data-testid="stButton"] button {
        background-color: #E50914;
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 4px;
        padding: 12px 28px;
        font-size: 16px;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #f6121d;
        color: white;
    }

    /* ---- Inputs ---- */
    div[data-testid="stTextInput"] input {
        background-color: #1f1f1f;
        color: #ffffff;
        border: 1px solid #444;
        border-radius: 4px;
        padding: 10px;
    }

    /* ---- Movie cards ---- */
    .movie-card {
        background: linear-gradient(135deg, #1f1f1f, #2c2c2c);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        border-left: 4px solid #E50914;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .movie-card:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 20px rgba(229, 9, 20, 0.4);
    }
    .movie-title { font-size: 18px; font-weight: 700; margin-bottom: 4px; }
    .movie-genre { font-size: 13px; color: #b3b3b3; margin-bottom: 6px; }
    .similarity-badge {
        display: inline-block;
        background-color: #E50914;
        color: white;
        font-size: 12px;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 12px;
    }
    .hero-card {
        background: linear-gradient(135deg, #E50914, #831010);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
    }
    .hero-title { font-size: 28px; font-weight: 800; }
    .hero-genre { font-size: 15px; color: #f5c6c6; }
    .catalog-badge {
        display: inline-block;
        background-color: #1f1f1f;
        color: #b3b3b3;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 13px;
    }
    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 24px;
    }
</style>
""", unsafe_allow_html=True)


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


# ---------- Session state ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

df, similarity_matrix = load_data()

# =========================================================
# LANDING / LOGIN PAGE
# =========================================================
if not st.session_state.logged_in:
    st.markdown(f"""
    <div class="hero-wrap">
        <div class="brand-logo">🎬 CINENEPAL</div>
        <div class="hero-headline">Unlimited Nepali movies,<br>tailored to your taste</div>
        <div class="hero-sub">{len(df)} movies. Smart recommendations. Completely free.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        name = st.text_input("", placeholder="Enter your name to get started", key="login_name")
        if st.button("Get Started ▶", use_container_width=True):
            if name.strip():
                st.session_state.logged_in = True
                st.session_state.username = name.strip()
                st.rerun()
            else:
                st.warning("Please enter your name to continue.")

# =========================================================
# MAIN APP (after "login")
# =========================================================
else:
    st.markdown(f"""
    <div class="topbar">
        <div style="font-size:24px; font-weight:800; color:#E50914;">🎬 CINENEPAL</div>
        <div class="catalog-badge">👋 {st.session_state.username} · {len(df)} movies in catalog</div>
    </div>
    """, unsafe_allow_html=True)

    query = st.text_input("", placeholder="🔍 Search for a movie you like (e.g. Kabaddi)")

    if query:
        clean_query = re.escape(query.lower().strip())
        matches = df[df["title"].str.lower().str.contains(clean_query, na=False)]

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

            genre_text = ', '.join(selected.genres) if selected.genres else 'Unknown'
            st.markdown(f"""
            <div class="hero-card">
                <div class="hero-title">▶ {selected.title}</div>
                <div class="hero-genre">{genre_text}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Because you liked this, you might enjoy:")

            recs = recommend_by_index(df, similarity_matrix, idx, top_n=5)
            recs["genres"] = recs["genres"].apply(lambda g: ", ".join(g) if g else "Unknown")

            cols = st.columns(3)
            for i, (_, row) in enumerate(recs.iterrows()):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="movie-title">{row.title}</div>
                        <div class="movie-genre">{row.genres}</div>
                        <span class="similarity-badge">{row.similarity_score:.0%} match</span>
                    </div>
                    """, unsafe_allow_html=True)

    st.write("")
    if st.button("Log out"):
        st.session_state.logged_in = False
        st.rerun()

import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# --------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Musically | Music Analytics & Recommendations",
    page_icon="🎶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling to match Musically theme
st.markdown("""
<style>
    .main-title {
        color: #8B0000;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .eyebrow {
        color: #C0392B;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }
    .subtext {
        color: #7f8c8d;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-container {
        background-color: #fde8e8;
        border: 1px solid #f5b7b1;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# CACHED DATA PIPELINE & MODEL TRAINING
# --------------------------------------------------
@st.cache_resource(show_spinner="Processing dataset and training PCA model...")
def load_and_process_data():
    dataset_path = "SpotifyFeatures.csv"
    if not os.path.exists(dataset_path):
        st.error(f"Dataset file '{dataset_path}' not found.")
        st.stop()

    df_raw = pd.read_csv(dataset_path)
    duplicate_count = int(df_raw.duplicated().sum())
    df = df_raw.drop_duplicates().reset_index(drop=True)

    audio_features = [
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "loudness",
        "speechiness",
        "tempo",
        "valence"
    ]

    features = [col for col in audio_features if col in df.columns]
    if len(features) < 2:
        st.error("The dataset must contain at least two usable audio features.")
        st.stop()

    audio_data = df[features].copy()
    for col in features:
        audio_data[col] = pd.to_numeric(audio_data[col], errors="coerce")

    audio_data = audio_data.replace([np.inf, -np.inf], np.nan)
    valid_rows = audio_data.notna().any(axis=1)

    df = df.loc[valid_rows].reset_index(drop=True)
    audio_data = audio_data.loc[valid_rows].reset_index(drop=True)

    non_constant_features = [
        col for col in features if audio_data[col].nunique(dropna=True) > 1
    ]
    audio_data = audio_data[non_constant_features]
    features = non_constant_features

    audio_data = audio_data.fillna(audio_data.median())

    # Standardize features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(audio_data)

    # PCA
    pca_all = PCA()
    pca_all.fit(x_scaled)
    cum_var = np.cumsum(pca_all.explained_variance_ratio_)

    n_components = int(np.argmax(cum_var >= 0.95) + 1)
    pca_model = PCA(n_components=n_components)
    reduced_data = pca_model.fit_transform(x_scaled)

    reconstructed_data = pca_model.inverse_transform(reduced_data)
    reconstruction_mse = mean_squared_error(x_scaled, reconstructed_data)
    variance_retained = float(np.sum(pca_model.explained_variance_ratio_) * 100)
    dim_reduction = ((len(features) - n_components) / len(features)) * 100

    # Silhouette score on a sample
    silhouette = None
    try:
        if len(reduced_data) >= 3:
            cluster_count = min(3, len(reduced_data) - 1)
            kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(reduced_data)
            if len(np.unique(cluster_labels)) > 1:
                silhouette = float(
                    silhouette_score(
                        reduced_data,
                        cluster_labels,
                        sample_size=min(5000, len(reduced_data)),
                        random_state=42
                    )
                )
    except Exception:
        silhouette = None

    # Song metadata
    songs = df.reset_index(drop=True).copy()
    track_col = next((c for c in ["track_name", "track", "name"] if c in songs.columns), None)
    artist_col = next((c for c in ["artist_name", "artist", "artists"] if c in songs.columns), None)
    genre_col = next((c for c in ["genre", "track_genre"] if c in songs.columns), None)
    pop_col = next((c for c in ["popularity", "track_popularity"] if c in songs.columns), None)

    if track_col is None:
        songs["track_name"] = [f"Song {i+1}" for i in range(len(songs))]
        track_col = "track_name"

    songs["display_track"] = songs[track_col].fillna("Unknown Song").astype(str)
    songs["display_artist"] = songs[artist_col].fillna("Unknown Artist").astype(str) if artist_col else "Unknown Artist"
    songs["display_genre"] = songs[genre_col].fillna("Unknown").astype(str) if genre_col else "Unknown"
    songs["display_popularity"] = (
        pd.to_numeric(songs[pop_col], errors="coerce").fillna(0).astype(int)
        if pop_col else 0
    )

    # Precompute curated popular songs for selectbox options
    genre_top = (
        songs.sort_values(by=["display_popularity", "display_track"], ascending=[False, True])
        .groupby("display_genre", as_index=False)
        .head(30)
    )
    overall_top = (
        songs.sort_values(by=["display_popularity", "display_track"], ascending=[False, True])
        .head(400)
    )
    curated_idx = (
        pd.concat([genre_top, overall_top])
        .drop_duplicates(subset=["display_track", "display_artist"])
        .index.tolist()
    )

    curated_df = songs.loc[curated_idx].copy()
    curated_df["orig_index"] = curated_df.index
    curated_df = curated_df.sort_values(
        by=["display_track", "display_artist"],
        key=lambda col: col.str.lower()
    )

    song_options = []
    for orig_idx, trk, art, gnr in zip(
        curated_df["orig_index"],
        curated_df["display_track"],
        curated_df["display_artist"],
        curated_df["display_genre"]
    ):
        song_options.append({
            "index": int(orig_idx),
            "label": f"{trk} — {art} ({gnr})"
        })

    return {
        "df": df,
        "audio_data": audio_data,
        "features": features,
        "songs": songs,
        "song_options": song_options,
        "reduced_data": reduced_data,
        "pca_model": pca_model,
        "cumulative_variance": cum_var,
        "n_components": n_components,
        "variance_retained": variance_retained,
        "reconstruction_mse": reconstruction_mse,
        "dim_reduction": dim_reduction,
        "silhouette": silhouette,
        "duplicates": duplicate_count,
        "has_popularity": pop_col is not None
    }


# Load data
data_bundle = load_and_process_data()
df = data_bundle["df"]
audio_data = data_bundle["audio_data"]
features = data_bundle["features"]
songs = data_bundle["songs"]
song_options = data_bundle["song_options"]
reduced_data = data_bundle["reduced_data"]
pca_model = data_bundle["pca_model"]
cumulative_variance = data_bundle["cumulative_variance"]
n_components = data_bundle["n_components"]
variance_retained = data_bundle["variance_retained"]
reconstruction_mse = data_bundle["reconstruction_mse"]
dim_reduction = data_bundle["dim_reduction"]
silhouette = data_bundle["silhouette"]
duplicate_count = data_bundle["duplicates"]
has_popularity = data_bundle["has_popularity"]


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.markdown("### 🎵 Musically")
st.sidebar.caption("Music Analytics & Recommendation")

page = st.sidebar.radio(
    "Navigation",
    ["Dataset Information", "EDA & Analysis", "Song Recommendations", "PCA & Performance"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset Size:** {:,} songs  \n**Features:** {} audio characteristics".format(
        len(df), len(features)
    )
)


# ==================================================
# PAGE 1: DATASET INFORMATION
# ==================================================
if page == "Dataset Information":
    st.markdown('<p class="eyebrow">DATA EXPLORATION</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">Spotify Dataset</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtext">Explore the structure, summary statistics, and characteristics of the music dataset.</p>', unsafe_allow_html=True)

    # Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tracks", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Audio Features", len(features))
    with col4:
        st.metric("Duplicate Rows", duplicate_count)

    st.markdown("---")

    # Sample Table
    st.subheader("Sample Records Preview")
    sample_preview = songs.head(10)[["display_track", "display_artist", "display_genre"]].rename(
        columns={"display_track": "Track", "display_artist": "Artist", "display_genre": "Genre"}
    )
    st.dataframe(sample_preview, use_container_width=True)

    # Column Information
    st.subheader("Column Metadata & Missing Values")
    col_info = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": [str(df[c].dtype) for c in df.columns],
        "Missing Values": [int(df[c].isna().sum()) for c in df.columns],
        "Unique Values": [int(df[c].nunique()) for c in df.columns]
    })
    st.dataframe(col_info, use_container_width=True)

    # Summary Statistics
    st.subheader("Statistical Summary of Audio Features")
    st.dataframe(audio_data.describe().round(2), use_container_width=True)


# ==================================================
# PAGE 2: EDA & FEATURE ANALYSIS
# ==================================================
elif page == "EDA & Analysis":
    st.markdown('<p class="eyebrow">MUSIC INSIGHTS</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">Exploratory Data Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtext">Understand distributions, correlations, and relationships among audio features.</p>', unsafe_allow_html=True)

    # Grid of charts
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.subheader("Feature Distributions")
        st.caption("Distribution across audio feature values.")
        dist_sample = audio_data.sample(min(20000, len(audio_data)), random_state=42)
        fig_dist, ax_dist = plt.subplots(figsize=(8, 6))
        dist_sample.hist(ax=ax_dist, bins=25, edgecolor="black")
        plt.tight_layout()
        st.pyplot(fig_dist)
        plt.close(fig_dist)

    with row1_col2:
        st.subheader("Correlation Heatmap")
        st.caption("Pearson correlation between audio features.")
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(audio_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax_corr)
        plt.tight_layout()
        st.pyplot(fig_corr)
        plt.close(fig_corr)

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.subheader("Feature Boxplots (Outlier Analysis)")
        st.caption("Distribution spread and outlier visualization (sample of 5,000 tracks).")
        box_sample = audio_data.sample(min(5000, len(audio_data)), random_state=42)
        fig_box, ax_box = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=box_sample, orient="h", ax=ax_box)
        ax_box.set_xlabel("Value")
        plt.tight_layout()
        st.pyplot(fig_box)
        plt.close(fig_box)

    with row2_col2:
        if {"danceability", "energy"}.issubset(audio_data.columns):
            st.subheader("Danceability vs Energy")
            st.caption("Scatter relationship between two core musical characteristics.")
            scatter_sample = audio_data[["danceability", "energy"]].sample(min(5000, len(audio_data)), random_state=42)
            fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
            ax_scatter.scatter(scatter_sample["danceability"], scatter_sample["energy"], alpha=0.35, color="#E74C3C")
            ax_scatter.set_xlabel("Danceability")
            ax_scatter.set_ylabel("Energy")
            plt.tight_layout()
            st.pyplot(fig_scatter)
            plt.close(fig_scatter)

    if has_popularity:
        st.subheader("Song Popularity Distribution")
        pop_sample = songs["display_popularity"].sample(min(20000, len(songs)), random_state=42)
        fig_pop, ax_pop = plt.subplots(figsize=(10, 4))
        ax_pop.hist(pop_sample, bins=25, edgecolor="black", color="#F1948A")
        ax_pop.set_xlabel("Popularity Score (0-100)")
        ax_pop.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig_pop)
        plt.close(fig_pop)

    st.subheader("Feature Summary Statistics")
    st.dataframe(audio_data.describe().round(2), use_container_width=True)


# ==================================================
# PAGE 3: SONG RECOMMENDATIONS
# ==================================================
elif page == "Song Recommendations":
    st.markdown('<p class="eyebrow">DISCOVER MUSIC</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">Song Recommendations</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtext">Find tracks with audio characteristics and acoustic profiles similar to your favorite track.</p>', unsafe_allow_html=True)

    with st.container():
        st.markdown("### Find Similar Songs")
        col_rec1, col_rec2 = st.columns([3, 1])

        labels = [opt["label"] for opt in song_options]
        indices = [opt["index"] for opt in song_options]

        with col_rec1:
            selected_label = st.selectbox(
                "Select a seed song (type to search):",
                options=labels,
                index=0,
                help="Start typing any song or artist name to instantly filter the list."
            )

        with col_rec2:
            num_recs = st.selectbox(
                "Number of recommendations:",
                options=[5, 10, 15, 20],
                index=0
            )

        find_button = st.button("🎵 Find Recommendations", type="primary", use_container_width=True)

    if find_button or "last_selected" in st.session_state:
        if find_button:
            st.session_state["last_selected"] = selected_label
            st.session_state["last_num_recs"] = num_recs

        curr_label = st.session_state.get("last_selected", selected_label)
        curr_num = st.session_state.get("last_num_recs", num_recs)

        selected_idx = indices[labels.index(curr_label)]

        # Calculate cosine similarity using PCA reduced features
        similarity_scores = cosine_similarity(
            reduced_data[selected_idx].reshape(1, -1),
            reduced_data
        )[0]

        similar_indices = np.argsort(similarity_scores)[::-1]
        similar_indices = [i for i in similar_indices if i != selected_idx]
        top_indices = similar_indices[:curr_num]

        rec_df = songs.iloc[top_indices][
            ["display_track", "display_artist", "display_genre", "display_popularity"]
        ].copy()

        rec_df["Match Score"] = [
            f"{round(float(similarity_scores[i]) * 100, 1)}%" for i in top_indices
        ]

        rec_df = rec_df.rename(
            columns={
                "display_track": "Song",
                "display_artist": "Artist",
                "display_genre": "Genre",
                "display_popularity": "Popularity"
            }
        )

        rec_df = rec_df[["Song", "Artist", "Genre", "Match Score", "Popularity"]]

        st.markdown("---")
        st.subheader("Recommended Songs")
        st.caption(f"Top {curr_num} tracks similar in acoustic profile to **{curr_label}**.")
        st.dataframe(rec_df.reset_index(drop=True), use_container_width=True)


# ==================================================
# PAGE 4: PCA & PERFORMANCE
# ==================================================
elif page == "PCA & Performance":
    st.markdown('<p class="eyebrow">MACHINE LEARNING</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">PCA & Model Performance</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtext">Explore dimensionality reduction performance, explained variance, and cluster separation.</p>', unsafe_allow_html=True)

    # Metrics
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Original Features", len(features))
    with m2:
        st.metric("Reduced Components", n_components)
    with m3:
        st.metric("Variance Retained", f"{round(variance_retained, 2)}%")
    with m4:
        st.metric("Reconstruction MSE", f"{reconstruction_mse:.5f}")
    with m5:
        st.metric("Reduction Ratio", f"{round(dim_reduction, 2)}%")
    with m6:
        st.metric("Silhouette Score", f"{silhouette:.4f}" if silhouette is not None else "N/A")

    st.markdown("---")

    col_pca1, col_pca2 = st.columns(2)

    with col_pca1:
        st.subheader("PCA 2D Visualization")
        st.caption("Projection of songs onto the first two principal components (5,000 track sample).")
        rng = np.random.default_rng(42)
        sample_pca_idx = rng.choice(len(reduced_data), size=min(5000, len(reduced_data)), replace=False)
        pca_pts = reduced_data[sample_pca_idx]

        fig_pca, ax_pca = plt.subplots(figsize=(8, 5))
        ax_pca.scatter(pca_pts[:, 0], pca_pts[:, 1], alpha=0.45, color="#C0392B")
        ax_pca.set_xlabel("Principal Component 1")
        ax_pca.set_ylabel("Principal Component 2")
        plt.tight_layout()
        st.pyplot(fig_pca)
        plt.close(fig_pca)

    with col_pca2:
        st.subheader("Cumulative Explained Variance")
        st.caption("Variance retained as the number of components increases.")
        components = np.arange(1, len(cumulative_variance) + 1)

        fig_var, ax_var = plt.subplots(figsize=(8, 5))
        ax_var.plot(components, cumulative_variance * 100, marker="o", color="#E74C3C")
        ax_var.axhline(y=95, linestyle="--", color="red", label="95% Threshold")
        ax_var.axvline(x=n_components, linestyle="--", color="blue", label=f"Selected: {n_components}")
        ax_var.set_xlabel("Number of Principal Components")
        ax_var.set_ylabel("Cumulative Variance (%)")
        ax_var.legend()
        plt.tight_layout()
        st.pyplot(fig_var)
        plt.close(fig_var)

    st.subheader("Explained Variance by Component")
    comp_df = pd.DataFrame({
        "Principal Component": [f"PC{i+1}" for i in range(len(pca_model.explained_variance_ratio_))],
        "Explained Variance (%)": [round(float(v * 100), 2) for v in pca_model.explained_variance_ratio_]
    })
    st.dataframe(comp_df, use_container_width=True)

    with st.expander("ℹ️ Model Performance Interpretation"):
        st.markdown("""
        - **Explained Variance:** Higher values mean more of the original data's variation is retained.
        - **Reconstruction MSE:** Lower values mean less information was lost during dimensionality reduction.
        - **Dimensionality Reduction:** Shows the percentage reduction in feature dimensions.
        - **Silhouette Score:** Evaluates K-Means cluster separation on the reduced features (values closer to 1 indicate better separation).
        """)

import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Electronics Recommender",
    page_icon="🛒",
    layout="centered"
)

# ── Load pre-trained artefacts ─────────────────────────────────────────────────
# We use @st.cache_resource so the pickles are only loaded ONCE,
# even when the user interacts with the app multiple times.
@st.cache_resource
def load_artefacts():
    with open('svd_matrix.pkl', 'rb') as f:
        svd_matrix = pickle.load(f)
    with open('product_list.pkl', 'rb') as f:
        product_list = pickle.load(f)
    return svd_matrix, product_list

svd_matrix, product_list = load_artefacts()

# ── Recommendation function ────────────────────────────────────────────────────
def get_recommendations(selected_product_id, top_n=5):
    """
    Given a product ID, return the top_n most similar products
    using cosine similarity on the SVD-compressed matrix.
    """
    # Find the row index of the selected product in our matrix
    product_index = product_list.index(selected_product_id)

    # Compute cosine similarity between the selected product and ALL products.
    # We pass a single row [product_index] vs the full matrix.
    similarity_scores = cosine_similarity(
        [svd_matrix[product_index]],
        svd_matrix
    ).flatten()

    # Sort by score descending, skip index 0 (the product itself)
    ranked_indices = similarity_scores.argsort()[::-1]
    top_indices = [i for i in ranked_indices if i != product_index][:top_n]

    # Build a list of (product_id, similarity_score) tuples
    results = [
        (product_list[i], round(similarity_scores[i], 4))
        for i in top_indices
    ]
    return results

# ── UI Layout ──────────────────────────────────────────────────────────────────
st.title("🛒 Amazon Electronics")
st.subheader("Collaborative Filtering Recommendation Engine")
st.markdown(
    """
    This app uses **Matrix Factorization (SVD)** trained on Amazon Electronics
    reviews to find products that are rated similarly by the same users.
    Select any product below to see the top 5 recommendations.
    """
)
st.divider()

# Dropdown — let the user pick any product from the trained list
selected_product = st.selectbox(
    label="Select a Product ID to get recommendations:",
    options=product_list,
    index=0,
    help="These are Amazon ASIN product identifiers from the Electronics dataset."
)

# Button to trigger the recommendation
if st.button("🔍 Find Similar Products", type="primary"):
    recommendations = get_recommendations(selected_product, top_n=5)

    st.success(f"Top 5 products similar to **{selected_product}**")

    # Display each recommendation as a clean metric card
    for rank, (product_id, score) in enumerate(recommendations, start=1):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{rank}.** `{product_id}`")
        with col2:
            st.metric(label="Similarity", value=score)

    st.divider()
    st.caption(
        "💡 Similarity is computed via cosine similarity on a 10-component "
        "TruncatedSVD embedding trained on users who rated ≥ 50 products."
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Portfolio project · Amazon ML Summer School Application · Built with Scikit-learn & Streamlit")

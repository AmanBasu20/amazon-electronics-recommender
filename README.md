# 🛒Amazon Electronics Recommender

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An unsupervised machine learning recommendation engine built on the **Amazon Electronics Reviews** dataset (Kaggle). The system applies **Matrix Factorization via Truncated SVD** to discover latent user-product interaction patterns and serves real-time recommendations through a **Streamlit** web interface.

---

## 📸 Demo

> Select any product ID from the dropdown → click "Find Similar Products" → get 5 cosine-similarity ranked recommendations instantly.

---

## 🏆 Key Achievements

- **Accomplished** reduction of a 7.8M-row raw ratings matrix to a dense, inference-ready 10-dimensional embedding **as measured by** 100% elimination of out-of-memory errors during matrix construction **by doing** a filtering step that retained only users with ≥ 50 product ratings, cutting the active user space by ~96% while preserving the highest-signal collaborative data.

- **Accomplished** capture of the dominant variance structure of the user-item interaction space **as measured by** the explained variance ratio reported by TruncatedSVD across 10 latent components **by doing** applying Truncated SVD on the transposed sparse CSR matrix, projecting ~7,000 products into a compact latent factor space where geometric distance encodes behavioural similarity.

- **Accomplished** sub-second recommendation inference at serving time **as measured by** real-time response in the deployed Streamlit app with no pre-computed similarity table required **by doing** computing on-the-fly cosine similarity between a single query product embedding and the full SVD matrix, a lightweight O(n·k) operation where k=10 factors.

---

## 🧠 Why SVD + Cosine Similarity? (The Math, Simply Explained)

This is a question worth understanding deeply, because the choice is deliberate — not default.

### The Core Problem: Extreme Sparsity

The Amazon Electronics dataset has millions of users and hundreds of thousands of products. If you imagine a giant table where rows are users and columns are products, most cells are **empty** (a user has only rated a tiny fraction of all products). This is called a **sparse matrix**, and it's the defining challenge of recommendation systems.

A typical sparsity level here is **> 99.9%** — meaning fewer than 1 in 1,000 cells has a real value.

### Why NOT Deep Learning?

It's a fair question. Deep learning (e.g., neural collaborative filtering, autoencoders) is powerful, but it has a specific weakness here:

| Factor | Deep Learning | SVD (Our Choice) |
|---|---|---|
| **Data requirement** | Needs dense, rich data to generalize | Works excellently on sparse matrices |
| **Training time** | Minutes to hours (GPU recommended) | Seconds to minutes (CPU only) |
| **Interpretability** | Black box — hard to explain | Latent factors have geometric meaning |
| **Overfitting risk** | High on sparse data without careful regularization | Low — SVD naturally finds global structure |
| **Serving complexity** | Requires a model server or ONNX runtime | Just load a numpy array and compute dot products |

For a sparse, tabular, unsupervised problem **without item metadata**, SVD is not a compromise — it is the mathematically correct tool.

### How SVD Works Here (Intuition)

Singular Value Decomposition factorises the User-Item matrix **M** into three matrices:

```
M  ≈  U  ×  Σ  ×  Vᵀ
```

- **U** — how much each *user* relates to each latent "taste dimension"
- **Σ** — how important each taste dimension is globally
- **Vᵀ** — how much each *product* relates to each latent taste dimension

By keeping only the top **10** latent dimensions (via `TruncatedSVD`), we discard noise (random one-off ratings) and keep signal (consistent behavioural patterns). Each product is now represented as a **10-number vector** — its "taste fingerprint."

### Why Cosine Similarity for Inference?

Once every product is a vector in 10-dimensional space, we measure similarity by the **angle** between vectors, not their length:

```
cosine_similarity(A, B) = (A · B) / (‖A‖ × ‖B‖)
```

- Score of **1.0** → products are rated identically by the same user segments (parallel vectors)
- Score of **0.0** → products have no overlapping audience (perpendicular vectors)

Cosine similarity is preferred over Euclidean distance here because it is **scale-invariant** — a product with 1,000 ratings and a product with 50 ratings can still be meaningfully compared based on *who* rated them, not *how many*.

---

## 🗂️ Project Structure

```
.
├── ecommerce_recommendation_engine.ipynb   # Training pipeline (Jupyter)
├── app.py                                  # Streamlit inference app
├── requirements.txt                        # Dependencies
├── svd_matrix.pkl                          # Trained product embeddings (generated)
├── product_list.pkl                        # Product ID index (generated)
└── ratings_Electronics.csv                # Raw dataset (not committed — download from Kaggle)
```

---

## ⚙️ Setup & Usage

### 1. Download the Dataset
Get `ratings_Electronics.csv` from [Kaggle — Amazon Product Reviews](https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews) and place it in the project root.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Training Notebook
Open and run all cells in `ecommerce_recommendation_engine.ipynb`.  
This will generate `svd_matrix.pkl` and `product_list.pkl`.

### 4. Launch the Web App
```bash
streamlit run app.py
```

---

## 🔧 Technical Stack

| Component | Library | Purpose |
|---|---|---|
| Data wrangling | `pandas` | Load, filter, pivot the ratings data |
| Sparse matrix | `scipy.sparse.csr_matrix` | Memory-efficient matrix storage |
| Dimensionality reduction | `sklearn.TruncatedSVD` | Latent factor extraction |
| Similarity metric | `sklearn.cosine_similarity` | Product-to-product scoring |
| Model persistence | `pickle` | Save/load trained artefacts |
| Web interface | `streamlit` | Interactive recommendation UI |

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | Amazon Product Reviews — Electronics (Kaggle) |
| Raw rows | ~7.8 million ratings |
| Columns | `user_id`, `product_id`, `rating`, `timestamp` |
| Filtering threshold | Users with ≥ 50 ratings retained |
| SVD components | 10 latent factors |

---

*"The simplest model that solves the problem correctly is the best model."*

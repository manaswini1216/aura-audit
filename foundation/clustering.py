"""
clustering.py
--------------
Phase 1: Foundation
Step 2: Unsupervised Issue Discovery using K-Means

- Loads preprocessed support logs
- Vectorizes text using TF-IDF
- Applies K-Means clustering
- Saves cluster assignments for downstream labeling

Author: Aura-Audit Intern Evaluation
"""

import random
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# -------------------------------
# Global configuration
# -------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

N_CLUSTERS = 5   # small, interpretable number for auditability


# -------------------------------
# Clustering pipeline
# -------------------------------
def run_clustering(
    input_path: str = "data/processed/clean_support_logs.csv",
    output_path: str = "data/processed/clustered_support_logs.csv",
):
    print("Loading cleaned data...")
    df = pd.read_csv(input_path)

    if "clean_text" not in df.columns:
        raise ValueError("Expected 'clean_text' column not found.")

    texts = df["clean_text"].tolist()

    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=3000,
        stop_words="english"
    )
    X = vectorizer.fit_transform(texts)

    print(f"Running K-Means with {N_CLUSTERS} clusters...")
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_SEED,
        n_init=10
    )
    cluster_labels = kmeans.fit_predict(X)

    df["cluster_id"] = cluster_labels

    print("Saving clustered data...")
    df.to_csv(output_path, index=False)

    print("Clustering completed successfully.")

    return df, kmeans, vectorizer


# -------------------------------
# Script entry point
# -------------------------------
if __name__ == "__main__":
    run_clustering()

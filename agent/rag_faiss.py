"""
rag_faiss.py
-------------
Phase 3: Agents & Governance
Step 10: Retrieval-Augmented Generation (RAG) with FAISS

- Builds a FAISS vector store from support logs
- Enables semantic retrieval of similar past issues
- Used by the ReAct agent as an external knowledge tool

Author: Aura-Audit Intern Evaluation
"""

import random
import numpy as np
import pandas as pd
import faiss

from sklearn.feature_extraction.text import TfidfVectorizer


# -------------------------------
# Global configuration
# -------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# -------------------------------
# FAISS RAG pipeline
# -------------------------------
class RAGVectorStore:
    def __init__(self, texts):
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            stop_words="english"
        )
        self.embeddings = self.vectorizer.fit_transform(texts).toarray()

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

        self.texts = texts

    def retrieve(self, query, top_k=3):
        query_vec = self.vectorizer.transform([query]).toarray()
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for idx in indices[0]:
            results.append(self.texts[idx])

        return results


# -------------------------------
# Build vector store
# -------------------------------
def build_rag_store(
    input_path: str = "data/processed/clean_support_logs.csv"
):
    print("Loading cleaned support logs...")
    df = pd.read_csv(input_path)

    if "clean_text" not in df.columns:
        raise ValueError("Expected 'clean_text' column not found.")

    texts = df["clean_text"].tolist()

    print("Building FAISS vector store...")
    store = RAGVectorStore(texts)

    print("RAG vector store ready.")
    return store


# -------------------------------
# Script entry point
# -------------------------------
if __name__ == "__main__":
    rag_store = build_rag_store()

    sample_query = "unable to login to my account"
    retrieved = rag_store.retrieve(sample_query)

    print("\n Retrieved similar logs:")
    for r in retrieved:
        print("-", r)

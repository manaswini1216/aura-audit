"""
mlp_model.py
------------
Phase 2: Neural & RL
Step 5: Neural Network Classifier (MLP)

- Trains a Multi-Layer Perceptron for intent classification
- Uses TF-IDF features for consistency
- Provides a neural baseline for agent usage

Author: Aura-Audit Intern Evaluation
"""

import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score


# -------------------------------
# Global configuration
# -------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# -------------------------------
# Training pipeline
# -------------------------------
def train_mlp(
    input_path: str = "data/processed/labeled_support_logs.csv",
):
    print("Loading labeled data...")
    df = pd.read_csv(input_path)

    if "clean_text" not in df.columns or "intent" not in df.columns:
        raise ValueError("Dataset must contain 'clean_text' and 'intent' columns.")

    X = df["clean_text"]
    y = df["intent"]

    print("Splitting train/test data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y
    )

    print("Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(
        max_features=3000,
        stop_words="english"
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training MLP classifier...")
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        max_iter=20,
        random_state=RANDOM_SEED,
        early_stopping=True
    )

    model.fit(X_train_vec, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n Accuracy: {acc:.4f}\n")
    print(" Classification Report:")
    print(classification_report(y_test, y_pred))

    return model, vectorizer


# -------------------------------
# Script entry point
# -------------------------------
if __name__ == "__main__":
    train_mlp()

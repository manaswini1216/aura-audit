"""
shap_explain.py
---------------
Phase 3: Governance
Step 8: Post-processing Audit using SHAP Explainability

- Generates SHAP explanations for model predictions
- Focuses on transparency and accountability
- Uses Random Forest baseline for interpretability

Author: Aura-Audit Intern Evaluation
"""

import random
import numpy as np
import pandas as pd
import shap

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# -------------------------------
# Global configuration
# -------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# -------------------------------
# SHAP explanation pipeline
# -------------------------------
def run_shap_explainer(
    input_path: str = "data/processed/labeled_support_logs.csv",
):
    print("Loading labeled data...")
    df = pd.read_csv(input_path)

    X = df["clean_text"]
    y = df["intent"]

    print("Splitting data...")
    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y
    )

    print("🔎 Vectorizing text...")
    vectorizer = TfidfVectorizer(
        max_features=2000,
        stop_words="english"
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Random Forest for explainability...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_SEED,
        class_weight="balanced"
    )
    model.fit(X_train_vec, y_train)

    print("Running SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_vec[:50])

    print("SHAP values computed successfully.")
    print("Use shap.summary_plot() in notebook for visualization.")

    return shap_values, explainer


# -------------------------------
# Script entry point
# -------------------------------
if __name__ == "__main__":
    run_shap_explainer()

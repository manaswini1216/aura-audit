"""
preprocess.py
--------------
Phase 1: Foundation
Step 1: Normalize Data & Remove PII

- Loads raw support logs
- Cleans text
- Removes personally identifiable information (PII)
- Saves processed data for downstream tasks

Author: Aura-Audit Intern Evaluation
"""

import re
import random
import numpy as np
import pandas as pd


# -------------------------------
# Global configuration
# -------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# -------------------------------
# PII removal utilities
# -------------------------------
def remove_pii(text: str) -> str:
    """
    Removes common PII patterns from text.
    Covers:
    - Email addresses
    - Phone numbers
    - Credit card / ID-like numbers
    """
    if not isinstance(text, str):
        return ""

    # Remove emails
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[EMAIL]", text)

    # Remove phone numbers (various formats)
    text = re.sub(r"\b\d{10}\b", "[PHONE]", text)
    text = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]", text)

    # Remove long numeric IDs (>= 12 digits)
    text = re.sub(r"\b\d{12,}\b", "[ID]", text)

    return text


# -------------------------------
# Text normalization
# -------------------------------
def clean_text(text: str) -> str:
    """
    Basic text normalization:
    - Lowercase
    - Remove special characters
    - Strip extra spaces
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -------------------------------
# Main preprocessing pipeline
# -------------------------------
def preprocess_data(
    input_path: str = "data/raw/support_logs.csv",
    output_path: str = "data/processed/clean_support_logs.csv",
):
    """
    End-to-end preprocessing pipeline.
    """

    print("Loading raw data...")
    df = pd.read_csv(input_path)

    text_col = None
    for col in df.columns:
        if "text" in col.lower() or "message" in col.lower() or "log" in col.lower():
            text_col = col
            break

    if text_col is None:
        raise ValueError("No suitable text column found in dataset.")

    print(f"Processing column: {text_col}")

    df["clean_text"] = (
        df[text_col]
        .astype(str)
        .apply(remove_pii)
        .apply(clean_text)
    )
    df = df[df["clean_text"].str.len() > 5].reset_index(drop=True)

    print(f"Cleaned samples: {len(df)}")

    print("Saving processed data...")
    df.to_csv(output_path, index=False)

    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    preprocess_data()

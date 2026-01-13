"""
bias_audit.py
-------------
Phase 2/3: Governance
Step 7: In-processing Bias Audit

- Checks class distribution in labeled dataset
- Computes sample weights to mitigate imbalance
- Outputs summary for audit & downstream training

Author: Aura-Audit Intern Evaluation
"""

import random
import numpy as np
import pandas as pd

from sklearn.utils.class_weight import compute_sample_weight


# -------------------------------
# Global configuration
# -------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# -------------------------------
# Bias audit & re-weighting
# -------------------------------
def compute_weights(input_path: str = "data/processed/labeled_support_logs.csv"):
    print(" Loading labeled dataset for bias audit...")
    df = pd.read_csv(input_path)

    if "intent" not in df.columns:
        raise ValueError("Dataset must contain 'intent' column.")

    print(" Checking class distribution:")
    class_counts = df["intent"].value_counts()
    print(class_counts)

    # Compute sample weights inversely proportional to class frequency
    sample_weights = compute_sample_weight(class_weight="balanced", y=df["intent"])
    df["sample_weight"] = sample_weights

    print("\n Sample weights summary:")
    print(df.groupby("intent")["sample_weight"].mean())

    # Save weighted dataset for downstream models
    output_path = "data/processed/labeled_support_logs_weighted.csv"
    df.to_csv(output_path, index=False)

    print(f" Weighted dataset saved to {output_path}")
    return df


# -------------------------------
# Script entry point
# -------------------------------
if __name__ == "__main__":
    compute_weights()

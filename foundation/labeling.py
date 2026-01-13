"""
labeling.py
------------
Phase 1: Foundation
Step 3: Semi-Supervised Label Propagation

- Maps discovered clusters to intent labels
- Propagates labels to all samples
- Produces labeled data for supervised learning

Author: Aura-Audit Intern Evaluation
"""

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
# Cluster → Intent mapping
# -------------------------------
def get_cluster_label_map():
    """
    Manually defined mapping based on cluster inspection.
    This simulates human-in-the-loop labeling.
    """

    return {
        0: "billing_issue",
        1: "login_problem",
        2: "technical_error",
        3: "account_management",
        4: "general_query",
    }


# -------------------------------
# Label propagation
# -------------------------------
def propagate_labels(
    input_path: str = "data/processed/clustered_support_logs.csv",
    output_path: str = "data/processed/labeled_support_logs.csv",
):
    print("Loading clustered data...")
    df = pd.read_csv(input_path)

    if "cluster_id" not in df.columns:
        raise ValueError("Expected 'cluster_id' column not found.")

    cluster_label_map = get_cluster_label_map()

    print("Propagating labels from clusters...")
    df["intent"] = df["cluster_id"].map(cluster_label_map)

    # Drop rows with unmapped clusters (safety)
    df = df.dropna(subset=["intent"]).reset_index(drop=True)

    print("Label distribution:")
    print(df["intent"].value_counts())

    print("Saving labeled data...")
    df.to_csv(output_path, index=False)

    print("Semi-supervised labeling completed successfully.")

    return df


# -------------------------------
# Script entry point
# -------------------------------
if __name__ == "__main__":
    propagate_labels()

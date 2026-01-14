"""
guardrails.py
--------------
Phase 3: Governance
Step 9: Post-processing Guardrails

- Applies safety and compliance rules on model outputs
- Prevents unsafe or low-confidence automated actions
- Acts as a final responsible AI layer

Author: Aura-Audit Intern Evaluation
"""

import random
import numpy as np


# -------------------------------
# Global configuration
# -------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CONFIDENCE_THRESHOLD = 0.60


# -------------------------------
# Guardrail logic
# -------------------------------
def apply_guardrails(predicted_intent, confidence, text):
    """
    Applies post-processing rules before taking action.

    Rules:
    - Low confidence → human escalation
    - Sensitive keywords → manual review
    """

    sensitive_keywords = [
        "legal",
        "refund lawsuit",
        "harassment",
        "data breach",
        "security issue",
    ]

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "final_action": "escalate_to_human",
            "reason": "Low model confidence"
        }

    for keyword in sensitive_keywords:
        if keyword in text.lower():
            return {
                "final_action": "manual_review",
                "reason": f"Sensitive keyword detected: {keyword}"
            }

    return {
        "final_action": "auto_handle",
        "intent": predicted_intent,
        "reason": "Passed all guardrails"
    }


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    sample_output = apply_guardrails(
        predicted_intent="billing_issue",
        confidence=0.72,
        text="I was charged twice for my subscription"
    )

    print("Guardrail Decision:")
    print(sample_output)

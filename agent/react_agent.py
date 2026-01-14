"""
react_agent.py
--------------
Phase 3: Agents & Governance
Step 11: ReAct Agent

- Combines reasoning (intent classification)
- Uses retrieval (RAG) for context
- Applies learned policy (Q-learning)
- Enforces guardrails before action

Author: Aura-Audit Intern Evaluation
"""

import random
import numpy as np

from governance.guardrails import apply_guardrails
from agent.rag_faiss import build_rag_store
from intelligence.q_learning import train_q_learning


# -------------------------------
# Global configuration
# -------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# -------------------------------
# Mock intent classifier (placeholder)
# -------------------------------
def predict_intent(text):
    """
    Placeholder for trained RF / MLP classifier.
    Returns (intent, confidence).
    """

    # Simple heuristic fallback
    if "bill" in text:
        return "billing_issue", 0.75
    if "login" in text or "password" in text:
        return "login_problem", 0.80
    if "error" in text or "crash" in text:
        return "technical_error", 0.70

    return "general_query", 0.65


# -------------------------------
# ReAct Agent
# -------------------------------
class ReActAgent:
    def __init__(self):
        print(" Initializing ReAct Agent...")
        self.rag_store = build_rag_store()
        self.q_agent = train_q_learning()

    def run(self, user_query: str):
        print("\n Thought: Understanding user intent...")
        intent, confidence = predict_intent(user_query)

        print(f" Predicted Intent: {intent} (confidence={confidence:.2f})")

        print("\n Action: Retrieving similar past logs...")
        retrieved_logs = self.rag_store.retrieve(user_query)

        print("\n Observation: Retrieved context")
        for log in retrieved_logs:
            print("-", log)

        print("\n Applying guardrails...")
        decision = apply_guardrails(
            predicted_intent=intent,
            confidence=confidence,
            text=user_query
        )

        print("\n Final Decision:")
        print(decision)

        return decision


# -------------------------------
# Script entry point
# -------------------------------
if __name__ == "__main__":
    agent = ReActAgent()

    query = "I cannot login to my account after password reset"
    agent.run(query)

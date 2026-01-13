"""
q_learning.py
-------------
Phase 2: Neural & RL
Step 6: Reward-based Decision Optimization using Q-Learning

- Learns optimal actions for support intents
- Uses a simple tabular Q-learning setup
- Designed for interpretability and auditability

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

ALPHA = 0.1      # learning rate
GAMMA = 0.9      # discount factor
EPSILON = 0.2    # exploration rate
EPISODES = 500


# -------------------------------
# Environment definition
# -------------------------------
INTENTS = [
    "billing_issue",
    "login_problem",
    "technical_error",
    "account_management",
    "general_query",
]

ACTIONS = [
    "route_to_billing",
    "reset_password",
    "technical_support",
    "account_update",
    "auto_reply",
]


# Reward matrix (intent → action)
REWARD_MATRIX = {
    ("billing_issue", "route_to_billing"): 5,
    ("login_problem", "reset_password"): 5,
    ("technical_error", "technical_support"): 5,
    ("account_management", "account_update"): 5,
    ("general_query", "auto_reply"): 3,
}


# -------------------------------
# Q-learning agent
# -------------------------------
class QLearningAgent:
    def __init__(self):
        self.q_table = np.zeros((len(INTENTS), len(ACTIONS)))

    def choose_action(self, state_idx):
        if random.random() < EPSILON:
            return random.randint(0, len(ACTIONS) - 1)
        return np.argmax(self.q_table[state_idx])

    def update(self, state_idx, action_idx, reward):
        best_future = np.max(self.q_table[state_idx])
        self.q_table[state_idx, action_idx] += ALPHA * (
            reward + GAMMA * best_future - self.q_table[state_idx, action_idx]
        )


# -------------------------------
# Training loop
# -------------------------------
def train_q_learning():
    agent = QLearningAgent()

    for _ in range(EPISODES):
        intent = random.choice(INTENTS)
        state_idx = INTENTS.index(intent)

        action_idx = agent.choose_action(state_idx)
        action = ACTIONS[action_idx]

        reward = REWARD_MATRIX.get((intent, action), -1)
        agent.update(state_idx, action_idx, reward)

    print("Q-Learning training completed.")
    print("\n Learned Q-Table:")
    for i, intent in enumerate(INTENTS):
        best_action = ACTIONS[np.argmax(agent.q_table[i])]
        print(f"{intent:20s} → {best_action}")

    return agent


# -------------------------------
# Script entry point
# -------------------------------
if __name__ == "__main__":
    train_q_learning()

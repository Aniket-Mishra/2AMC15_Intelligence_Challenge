# agents/monte_carlo_agent.py

import numpy as np
from agents.base_agent import BaseAgent
from collections import defaultdict
import random

class MonteCarloAgent(BaseAgent):
    def __init__(self, action_space, epsilon=0.1, gamma=0.99):
        self.action_space = action_space  # typically [0, 1, 2, 3] for up/down/left/right
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(len(action_space)))
        self.visit_counts = defaultdict(lambda: np.zeros(len(action_space)))  # <-- new
        self.episode = []

    def take_action(self, state):
        """Select action using ε-greedy policy"""
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        state_key = self._state_to_key(state)
        return int(np.argmax(self.Q[state_key]))

    def update(self, state, action, reward, next_state, done):
        """Store the full episode to use for MC update after episode ends"""
        self.episode.append((state, action, reward))
        if done:
            self._update_Q()
            self.episode.clear()


    def _update_Q(self):
        G = 0
        visited = set()
        for t in reversed(range(len(self.episode))):
            state, action, reward = self.episode[t]
            G = self.gamma * G + reward
            state_key = self._state_to_key(state)

            if (state_key, action) not in visited:
                visited.add((state_key, action))

                # Increment visit count
                self.visit_counts[state_key][action] += 1
                n = self.visit_counts[state_key][action]

                # Incremental average update
                q_old = self.Q[state_key][action]
                self.Q[state_key][action] += (G - q_old) / n


    def _state_to_key(self, state):
        """Convert state to a hashable format (e.g., tuple if it's a position)"""
        return tuple(state) if isinstance(state, (list, np.ndarray)) else state

    @property
    def q_table(self):
        return self.Q
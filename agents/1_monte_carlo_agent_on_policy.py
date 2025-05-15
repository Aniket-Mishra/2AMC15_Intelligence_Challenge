# agents/monte_carlo_agent.py

import numpy as np
from agents.base_agent import BaseAgent
from collections import defaultdict
import random

class MonteCarloAgent(BaseAgent):
    def __init__(self, 
                 action_space, 
                 gamma=0.99,
                 alpha: float = 0.1,
                 alpha_decay: float = 0.9,
                 alpha_min: float = 0.01,
                 epsilon: float = 0.5,
                 epsilon_decay: float = 0.9,
                 epsilon_min: float = 0.01, 
                ):
        self.action_space = action_space  # typically [0, 1, 2, 3] for up/down/left/right
        self.alpha = alpha  # NEW: step size as parameter instead of always 1 / visit_count
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
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
                #self.visit_counts[state_key][action] += 1
                #n = self.visit_counts[state_key][action]

                # Incremental average update
                q_old = self.Q[state_key][action]
                #self.Q[state_key][action] += (G - q_old) / n
                self.Q[state_key][action] += self.alpha*(G - q_old) # Now update with custom step size


    def _state_to_key(self, state):
        """Convert state to a hashable format (e.g., tuple if it's a position)"""
        return tuple(state) if isinstance(state, (list, np.ndarray)) else state

    @property
    def q_table(self):
        return self.Q
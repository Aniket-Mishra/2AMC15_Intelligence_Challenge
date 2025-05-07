import numpy as np
from agents import BaseAgent

class QLearningAgent_2(BaseAgent):
    def __init__(self,
                 alpha: float = 0.1,
                 gamma: float = 0.9,
                 epsilon: float = 0.5,
                 epsilon_decay: float = 0.95,
                 epsilon_min: float = 0.01,
                 num_actions = [0, 1, 2, 3]):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_actions = num_actions
        self.q_table = {}


    def take_action(self, state: tuple[int, int]) -> int:
        if state not in self.q_table:
            self.q_table[state] = np.array([0.0 for _ in self.num_actions])
            
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            action = int(np.argmax(self.q_table[state]))
        # print(action) # Way too many prints
        return action

    def update(self, state: tuple[int, int], next_state: tuple[int, int], reward: float, action: int):
        if state not in self.q_table:
            self.q_table[state] = np.array([0.0 for _ in self.num_actions])
            
        if next_state not in self.q_table:
            self.q_table[next_state] = np.array([0.0 for _ in self.num_actions])
                
        current_q = self.q_table[state][action]
        max_future_q = np.max(self.q_table[next_state])

        # Q-learning update
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

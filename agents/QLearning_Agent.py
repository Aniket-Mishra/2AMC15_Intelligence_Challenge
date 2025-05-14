import numpy as np
from agents import BaseAgent

class QLearningAgent(BaseAgent):
    def __init__(self,
                 alpha: float = 0.1,
                 gamma: float = 0.9,
                 epsilon: float = 0.5,
                 epsilon_decay: float = 0.9,
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

    def _initialize_state(self, state: tuple[int, int]):
        """Ensure state exists in Q-table."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.num_actions), dtype=float)

    def take_action(self, state: tuple[int, int]) -> int:
        '''Take a random action with probability epsilon, 
        or the best known action for the given state with probability 1 - epsilon.'''
        self._initialize_state(state)
            
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            action = int(np.argmax(self.q_table[state]))

        return action

    def update(self, ep_decay: bool, state: tuple[int, int], next_state: tuple[int, int], reward: float, action: int):
        '''Update the Q-table according to the Q-learning update rule.'''
        self._initialize_state(state)
        self._initialize_state(next_state)
                
        current_q = self.q_table[state][action]
        max_future_q = np.max(self.q_table[next_state])

        # Q-learning update
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)

        # Decay epsilon to minimize exploration and increase exploitation in later stages
        if ep_decay: 
            self.epsilon *= self.epsilon_decay

    def eval_mode(self):
        '''Sets the agent to evaluation mode (call after training).
        This sets epsilon to 0, so the agent only takes the best action and doesn't explore.'''
        self.epsilon = 0        

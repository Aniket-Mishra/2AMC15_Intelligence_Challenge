### Q-LEARNING AGENT ###

import numpy as np
from random import random, randint

from agents import BaseAgent


class QLearningAgent(BaseAgent):
    """Agent that learns the optimal policy using Q-learning."""
    def __init__(self, alpha0: float, epsilon0: float, gamma: float, Q_default: float = 0.):
        '''
        Initializes the Q-learning agent.
        Note: for now alpha and epsilon are constant; the parameters are called alpha0 and epsilon0
            in case we want to include some sort of learning rate decay later on.
        Note: I'm not actually sure if the agent is allowed to know the environment (state space) in advance;
            I'll assume not. In that case, Q is an initially empty dictionary, to which states will be added as we encounter them.
            If a (s,a) pair does not exist in the Q-table, a lookup will return some default value, and an update will add the (s,a) pair.

        Parameters
        ---------------
        alpha0 : float > 0
            The initial SGD step size.
        epsilon0 : float in [0,1]
            The initial probability of taking a random action instead of the best so far.
        gamma : float in [0,1]
            The discount factor. Higher gamma means future rewards will be deemed less important.
        Q_default : float, default 0
            The default value for Q in case a state has not yet been visited.
        '''
        self.Q = {}
        self.curr_state = None
        self.alpha = alpha0
        self.epsilon = epsilon0
        self.gamma = gamma
        self.Q_default = Q_default

    def update(self, state: tuple[int, int], reward: float, action: int):
        '''Update the Q-table according to the Q-learning update rule.'''
        self.Q[(self.curr_state, action)] = (self.get_Q(self.curr_state, action) 
            + self.alpha * (reward + self.gamma * self.get_best_value(state) - self.get_Q(self.curr_state, action)))

    def take_action(self, state: tuple[int, int]) -> int:
        '''Take a random action with probability epsilon, 
        or the best known action for the given state with probability 1 - epsilon.'''
        self.curr_state = state  # Remember in what state we took this action (for use with the update function)
        # Note: this assumes that take_action and update will always be called alternatingly (but this is reasonable)
        if random() < self.epsilon:
            return randint(0,3)
        else:
            return self.get_best_action(state)
        
    def eval_mode(self):
        '''Sets the agent to evaluation mode (call after training).
        This sets epsilon to 0, so the agent only takes the best action and doesn't explore.
        If necessary, other stuff could be added here as well.'''
        self.epsilon = 0
        
    def get_Q(self, state: tuple[int, int], action) -> float:
        '''Helper function; returns Q[state, action] if the given state-action pair 
        has been visited before, or Q_default otherwise.'''
        try:
            return self.Q[(state, action)]
        except:
            return self.Q_default
        
    def get_best_action(self, state: tuple[int, int]) -> int:
        '''Helper function; gets the best known action for the given state.
        Note: the action space is currently hardcoded to be [0,1,2,3], but I assume this is fine.'''
        action_Qs = [self.get_Q(state, a) for a in range(4)]  # Q-values for all state-action pairs with the given state
        return np.argmax(action_Qs)
    
    def get_best_value(self, state: tuple[int, int]) -> int:
        '''Helper function; gets the value of the best known action for the given state.
        Note: the action space is currently hardcoded to be [0,1,2,3], but I assume this is fine.'''
        action_Qs = [self.get_Q(state, a) for a in range(4)]  # Q-values for all state-action pairs with the given state
        return np.max(action_Qs)

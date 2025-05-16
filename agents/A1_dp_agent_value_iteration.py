#Sources: https://www.geeksforgeeks.org/implement-value-iteration-in-python/
from agents import BaseAgent
from world import Environment


def run_value_iteration(env: Environment, gamma: float, theta: float):
    # Initially set V to 0 for all states
    V = {s: 0.0 for s in env.get_state_space()}

    deltas: list[float] = []
    iteration = 0
    values = []
    mean_values = []

    start_state = env.reset()

    while True:
        # Store delta to exit the loop when the change in value is lower than theta
        delta = 0
        for s in V:
            # Get action values for a state
            action_values = get_action_values(s, V, env, gamma)
            max_q = max(action_values)
            delta = max(delta, abs(V[s] - max_q))
            # Bellman optimality
            V[s] = max_q
        # Track the maximum change in delta
        deltas.append(delta)
        iteration += 1
        values.append(V[start_state])
        mean_values.append(sum(V.values()) / len(V))

        if delta < theta:
            break

    # Extract greedy policy
    policy = extract_greedy_policy(V, env, gamma)

    # package metrics
    metrics = {
        "iterations": iteration,
        "steps_taken": 0,
        "deltas": deltas,
        "values": values,
        "mean_values": mean_values
    }

    return V, policy, metrics


def get_action_values(s: tuple[int, int], V: dict, env: Environment, gamma: float):
    action_values = []
    for a in env.get_action_space():
        q = 0
        for s_next, p, r in env.get_transition_model(s, a):
            q += p * (r + gamma * V[s_next])
        action_values.append(q)
    return action_values


def extract_greedy_policy(V: dict, env: Environment, gamma: float):
    policy = {}
    for s in V:
        best_a = None
        best_q = float("-inf")
        for a in env.get_action_space():
            q = sum(p * (r + gamma * V[s_next]) for (s_next, p, r) in env.get_transition_model(s, a))
            if q > best_q:
                best_q, best_a = q, a
        policy[s] = best_a

    return policy


class ValueIterationAgent(BaseAgent):
    def __init__(self, env, gamma=0.99, theta=1e-6):
        """
        Runs value iteration on the current environment setup to produce an optimal policy
        """
        super().__init__()
        self.V, self.policy, self.metrics = run_value_iteration(env, gamma, theta)

    def take_action(self, observation):
        """
        Take action depending on the current state of the agent
        """
        return self.policy[observation]

    def update(self, *args, **kwargs):
        """
        No updates needed because DP is offline, so no online learning
        """
        pass

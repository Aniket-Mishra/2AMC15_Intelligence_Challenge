#Sources: https://www.geeksforgeeks.org/implement-value-iteration-in-python/

def run_value_iteration(env, gamma=0.99, theta=1e-6):
    # Initially set V to 0 for all states
    V = {s: 0.0 for s in env.get_state_space()}

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
        if delta < theta:
            break

    # Extract greedy policy
    policy = extract_greedy_policy(V, env, gamma)

    return V, policy


def get_action_values(s, V, env, gamma):
    action_values = []
    for a in env.get_action_space():
        q = 0
        for s_next, p, r in env.get_transition_model(s, a):
            q += p * (r + gamma * V[s_next])
        action_values.append(q)
    return action_values


def extract_greedy_policy(V, env, gamma):
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




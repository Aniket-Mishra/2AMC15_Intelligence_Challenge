
def custom_reward_function(grid, agent_pos) -> float:
    """This is a very simple custom reward function which follows
    the same signature, as the default reward function.

    Args:
        grid: The grid the agent is moving on, in case that is needed by
            the reward function.
        agent_pos: The position the agent is moving to.

    Returns:
        A single floating point value representing the reward for a given
        action.
    """

    match grid[agent_pos]:
        case 0:  # Moved to an empty tile
            reward = -1
        case 1 | 2:  # Moved to a wall or obstacle
            reward = -7
        case 3:  # Moved to a target tile
            reward = 50
            # "Illegal move"
        case _:
            raise ValueError(f"Grid cell should not have value: {grid[agent_pos]}.",
                                f"at position {agent_pos}")
    return reward
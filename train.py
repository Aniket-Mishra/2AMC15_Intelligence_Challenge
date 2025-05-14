"""
Train your RL Agent in this file. 
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
from world.reward_functions import custom_reward_function
from world.helpers import action_to_direction
from agents.monte_carlo_agent_on_policy import MonteCarloAgent

try:
    from world import Environment
    from agents.random_agent import RandomAgent
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys
    root_path = path.abspath(path.join(
        path.join(path.abspath(__file__), pardir), pardir)
    )
    if root_path not in sys.path:
        sys.path.extend(root_path)
    from world import Environment
    from agents.random_agent import RandomAgent

def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than "
                        "one.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=-1,
                   help="Frames per second to render at. Only used if "
                        "no_gui is not set.")
    p.add_argument("--iter", type=int, default=1000,
                   help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    return p.parse_args()


def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, random_seed: int, agent_start_pos: tuple[int,int]):
    """Main loop of the program."""

    for grid in grid_paths:
        # Set up the environment
        env = Environment(grid, no_gui, sigma=sigma, agent_start_pos=agent_start_pos, reward_fn=custom_reward_function, target_fps=fps, 
                          random_seed=random_seed)
        
        state = env.reset()
        
        # The model
        print(f"\nTransition model at start‐state {state!r}:")
        for a in env.get_action_space():
            print(f" Action {a!r} ({action_to_direction(a)}):")
            for (s_prime, prob, rew) in env.get_transition_model(state, a):
                print(f"   → next={s_prime!r},  p={prob:.6f},  r={rew}")

        
        # Initialize agent
        agent = MonteCarloAgent(action_space=[0, 1, 2, 3], epsilon=0.1, gamma=0.99)

        print(f"\nTraining on grid: {grid.name}")

        # Epsilon decay parameters
        initial_epsilon = 1.0
        final_epsilon = 0.5
        decay_rate = 0.995

        
        for episode in trange(iters, desc="Episodes"):
            # Decay epsilon
            agent.epsilon = max(final_epsilon, initial_epsilon * (decay_rate ** episode))

            state = env.reset()
            done = False

            while not done:
                action = agent.take_action(state)
                next_state, reward, done, info = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state  # move to next state


        # Evaluate the agent
        agent.epsilon = 0.0  # Greedy evaluation policy
        Environment.evaluate_agent(grid, agent, iters, sigma=0, agent_start_pos=agent_start_pos, reward_fn=custom_reward_function, random_seed=random_seed)


if __name__ == '__main__':
    args = parse_args()
    #Hard-coded --> Set the initial starting position for both training and evaluating
    agent_start_pos=(1, 1)
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.random_seed, agent_start_pos)
"""
Train your RL Agent in this file. 
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange

try:
    from world import Environment
    from agents.random_agent import RandomAgent
    from agents.Q_learning_agent import QLearningAgent
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
    from agents.Q_learning_agent import QLearningAgent

def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than "
                        "one.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if "
                        "no_gui is not set.")
    p.add_argument("--episodes", type=int, default=1000,
                   help="Number of episodes to go through.")
    p.add_argument("--iter", type=int, default=200,
                   help="Max number of iterations to go through per episode.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    return p.parse_args()


def main(grid_paths: list[Path], no_gui: bool, episodes: int, iters: int, fps: int,
         sigma: float, random_seed: int, agent_start_pos: tuple[int,int]):
    """Main loop of the program."""

    for grid in grid_paths:
        
        # Set up the environment
        env = Environment(grid, no_gui,sigma=sigma, target_fps=fps, 
                          agent_start_pos=agent_start_pos, random_seed=random_seed)
        
        # Initialize agent
        #agent = RandomAgent()
        agent = QLearningAgent(alpha0 = 0.2, epsilon0 = 0.2, gamma = 0.9)
        
        # Always reset the environment to initial state at the start of each episode
        for _ in trange(episodes):
            state = env.reset()
            for _ in trange(iters):
                
                # Agent takes an action based on the latest observation and info.
                action = agent.take_action(state)

                # The action is performed in the environment
                state, reward, terminated, info = env.step(action)
                
                # If the final state is reached, stop.
                if terminated:
                    break

                agent.update(state, reward, info["actual_action"])

        # Evaluate the agent
        Environment.evaluate_agent(grid, agent, iters, sigma, random_seed=random_seed, agent_start_pos=agent_start_pos)


if __name__ == '__main__':
    args = parse_args()
    agent_start_pos = [1, 13] # Hardcoded for now
    main(args.GRID, args.no_gui, args.episodes, args.iter, args.fps, args.sigma, args.random_seed, agent_start_pos)
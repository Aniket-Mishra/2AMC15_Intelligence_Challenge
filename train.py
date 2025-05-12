import importlib
import inspect
import os, json, datetime
from inspect import Parameter
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tqdm import trange
from typing import Any, Tuple
import numpy as np

from world.reward_functions import custom_reward_function
from world.helpers import action_to_direction
from world import Environment
from agents import BaseAgent


def parse_args() -> Namespace:
    p = ArgumentParser(description="DIC RL Agent Trainer")
    p.add_argument("GRID", type=Path, nargs="+", help="Path(s) to grid file(s)")
    p.add_argument("--agent", type=str, default="RandomAgent", help="Name of the agent to use")
    p.add_argument("--no_gui", action="store_true", help="Disable GUI rendering")
    p.add_argument("--sigma", type=float, default=0.1, help="Environment stochasticity (sigma)")
    p.add_argument("--fps", type=int, default=30, help="Frames per second for GUI")
    p.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    p.add_argument("--iter", type=int, default=200, help="Max iterations per episode")
    p.add_argument("--random_seed", type=int, default=0, help="Random seed")
    p.add_argument("--agent_start_pos", nargs=2, type=int, default=[1, 1], help="Start pos of agent")
    return p.parse_args()


def load_agent(agent_name: str, env: Environment) -> Tuple[BaseAgent, str]:
    with open("agent_config.json", "r") as f:
        config = json.load(f)
    if agent_name not in config:
        raise ValueError(f"Agent '{agent_name}' not found in config.")
    
    agent_info = config[agent_name]
    module = importlib.import_module(agent_info["module"])
    AgentClass = getattr(module, agent_info["class"])
    init_args = agent_info.get("init_args", {})

    sig = inspect.signature(AgentClass.__init__)
    if 'env' in sig.parameters:
        agent = AgentClass(env=env, **init_args)
    else:
        agent = AgentClass(**init_args)

    return agent, agent_info["train_mode"]


def update_agent(agent: BaseAgent, args: Namespace,
                        state: tuple[int, int],
                        next_state: tuple[int, int],
                        reward: float,
                        actual_action: int) -> None:
    update_params = inspect.signature(agent.update).parameters
    update_param_names = list(update_params)

    if {"state", "next_state"}.issubset(update_param_names):
        agent.update(state=state, next_state=next_state, reward=reward, action=actual_action)
    elif {"next_state", "reward", "action"}.issubset(update_param_names):
        agent.update(next_state=next_state, reward=reward, action=actual_action)
    elif {"state", "reward", "action"}.issubset(update_param_names):
        agent.update(state=state, reward=reward, action=actual_action)
    elif all(p.kind in {Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD} for p in update_params.values()):
        agent.update()
    else:
        raise ValueError(f"Agent '{args.agent}' has an unsupported update() signature: {update_param_names}")


def main(args: Namespace) -> None:
    start_pos: Tuple[int, int] = tuple(args.agent_start_pos)

    for grid in args.GRID:
        env = Environment(
            grid, args.no_gui, sigma=args.sigma, agent_start_pos=start_pos,
            reward_fn=custom_reward_function, target_fps=args.fps, random_seed=args.random_seed
        )
        env.reset()
        agent, mode = load_agent(args.agent, env)

        if mode == "episodic":
            #Max difference for convergence check
            metrics = {"iterations": 0, "deltas": [], "rewards": []}
            delta = 1e-7
            
            for ep in trange(args.episodes, desc=f"Training {args.agent}"):
                # Save a copy of the current Q-table for convergence check
                prev_q_table = {
                    s: np.copy(q_values) for s, q_values in agent.q_table.items()
                }
                state = env.reset()
                ep_reward = 0.0
                for _ in range(args.iter):
                    action = agent.take_action(state)
                    next_state, reward, terminated, info = env.step(action)
                    ep_reward += reward
                    if terminated:
                        break
                    #only start decaying epsilon after a number of episodes have passed
                    if ep < args.episodes/10:
                        ep_decay = False
                    else:
                        ep_decay = True
                    agent.update(ep_decay, state, next_state, reward, info["actual_action"])
                    state = next_state
                # # Convergence check
                common_states = set(agent.q_table.keys()) & set(prev_q_table.keys())
                if not common_states:
                    max_diff = 1
                else:
                    max_diff = max(
                        np.max(np.abs(agent.q_table[s] - prev_q_table[s]))
                        for s in common_states
                    )
                metrics["deltas"].append(max_diff)
                metrics["rewards"].append(ep_reward)
                # Stopping criterion
                if max_diff < delta:
                    metrics["iterations"] = ep
                    break

            if metrics["iterations"] == 0:
                metrics["iterations"] = args.episodes

            agent.metrics = metrics
    
            # Set epsilon to 0 so the agent always uses the best action
            agent.eval_mode()


        elif mode == "iterative":
            state = env.reset()
            for _ in trange(args.iter, desc=f"Training {args.agent}"):
                action = agent.take_action(state)
                next_state, reward, terminated, info = env.step(action)

                update_agent(agent, args, state, next_state, reward, info["actual_action"])

                state = next_state
                if terminated:
                    break

        else:
            raise ValueError(f"Unknown training mode '{mode}' for agent '{args.agent}'")

        if hasattr(agent, "metrics"):
            its = agent.metrics.get("iterations", None)
            print(f"[Metrics] {args.agent} converged in {its} iterations")
            metrics_dir = "metrics"
            os.makedirs(metrics_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{args.agent}_{ts}.json"
            path = os.path.join(metrics_dir, fname)
        try:
            with open(path, "w") as mf:
                json.dump(agent.metrics, mf, indent=2)
                print(f"[Metrics] Saved convergence data to {path}")
        except Exception as e:
            print(f"[Metrics] ERROR saving metrics: {e}")

        Environment.evaluate_agent(
            grid, agent, args.iter, args.sigma, agent_start_pos=start_pos,
            reward_fn=custom_reward_function, random_seed=args.random_seed
        )


if __name__ == "__main__":
    args: Namespace = parse_args()
    main(args)

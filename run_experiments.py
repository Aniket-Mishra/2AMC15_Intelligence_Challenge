import json
import uuid
from copy import deepcopy
import pandas as pd
import json, io, sys, re, importlib, inspect
from copy import deepcopy
from pathlib import Path
from argparse import Namespace
import numpy as np
import pandas as pd
from tqdm import trange

from world.reward_functions import custom_reward_function
from world import Environment
from agents import BaseAgent

import os, json, datetime, random


# Setting up train.py functions to run experiments

def load_agent(agent_name: str, env: Environment, config: dict):
    info = config[agent_name]
    mod  = importlib.import_module(info["module"])
    cls  = getattr(mod, info["class"])
    init_args = info.get("init_args", {})
    sig = inspect.signature(cls.__init__)
    if 'env' in sig.parameters:
        return cls(env=env, **init_args), info["train_mode"], info["init_args"]
    else:
        return cls(**init_args), info["train_mode"], info["init_args"]

def update_agent(agent: BaseAgent, args: Namespace, state, next_state, reward, action):
    params = inspect.signature(agent.update).parameters
    names  = set(params)
    if {"state","next_state"}.issubset(names):
        agent.update(state=state, next_state=next_state, reward=reward, action=action)
    elif {"next_state","reward","action"}.issubset(names):
        agent.update(next_state=next_state, reward=reward, action=action)
    elif {"state","reward","action"}.issubset(names):
        agent.update(state=state, reward=reward, action=action)
    else:
        agent.update()

def train_and_eval(args: Namespace, config: dict):
    start = tuple(args.agent_start_pos)

    for grid_fp in args.GRID:
        env = Environment(
            Path(grid_fp),
            args.no_gui,
            sigma=args.sigma,
            agent_start_pos=start,
            reward_fn=custom_reward_function,
            target_fps=args.fps,
            random_seed=args.random_seed
        )
        env.reset()
        agent, mode, init_args = load_agent(args.agent, env, config)

        if mode == "q_learning":
            #Max difference for convergence check
            metrics = {"iterations": 0, "steps_taken": 0, "deltas": [], "rewards": []}
            delta = 1e-6

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
                    agent.update(state, next_state, reward, info["actual_action"])
                    state = next_state

                if ep >= args.episodes/4:
                    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
                    agent.alpha = max(agent.alpha_min, agent.alpha * agent.alpha_decay)

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
                #metrics["steps_taken"] = env.world_stats["total_steps"]

                if max_diff < delta:
                    metrics["iterations"] = ep
                    break

            if metrics["iterations"] == 0:
                metrics["iterations"] = args.episodes

            agent.metrics = metrics

            agent.eval_mode()

        elif mode == "value_iteration":
            state = env.reset()
            for _ in trange(args.iter, desc=f"[Train] {args.agent}"):
                a  = agent.take_action(state)
                ns, r, done, info = env.step(a)
                update_agent(agent, args, state, ns, r, info["actual_action"])
                state = ns
                if done: break
            #agent.metrics["steps_taken"] = env.world_stats["total_steps"]

        elif mode == "monte_carlo":
            delta = 1e-6

            metrics = {"iterations": 0, "steps_taken": 0, "deltas": [], "rewards": []}

            for episode in trange(args.episodes, desc=f"Training {args.agent}"):
                prev_q = {s: np.copy(agent.q_table[s]) for s in agent.q_table}

                state = env.reset()
                terminated = False
                ep_reward = 0.0
                for _ in range(args.iter):
                    action = agent.take_action(state)
                    next_state, reward, terminated, info = env.step(action)
                    ep_reward += reward
                    if terminated:
                        break
                    agent.update(state, action, reward, next_state, False)
                    state = next_state

                agent.update(state, action, reward, next_state, True)

                if episode >= args.episodes/4:
                    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
                    agent.alpha = max(agent.alpha_min, agent.alpha * agent.alpha_decay)

                # Convergence check
                common_states = set(agent.q_table.keys()) & set(prev_q.keys())
                if not common_states:
                    max_diff = 1
                else:
                    max_diff = max(
                        np.max(np.abs(agent.q_table[s] - prev_q[s]))
                        for s in common_states
                    )

                metrics["deltas"].append(max_diff)
                metrics["rewards"].append(ep_reward)

                if max_diff < delta:
                    metrics["iterations"] = episode
                    break

            if metrics["iterations"] == 0:
                metrics["iterations"] = args.episodes

            agent.metrics = metrics
            agent.epsilon = 0.0  # Switch to greedy

        else:  # iterative / random
            state = env.reset()
            for _ in trange(args.iter, desc=f"[Train] {args.agent}"):
                a = agent.take_action(state)
                ns, r, done, info = env.step(a)
                update_agent(agent, args, state, ns, r, info["actual_action"])
                state = ns
                if done: break

        # capture evaluation output
        buf = io.StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = buf, buf
        try:
            no_steps = Environment.evaluate_agent(
                Path(args.GRID[0]),
                agent,
                args.iter,
                sigma=0.0, # No noise during evaluation
                agent_start_pos=start,
                reward_fn=custom_reward_function,
                random_seed=args.random_seed,
                show_images=False
            )
            agent.metrics["steps_taken"] = no_steps
        finally:
            sys.stdout, sys.stderr = old_out, old_err

        if hasattr(agent, "metrics"):
            its = agent.metrics.get("iterations", None)
            print(f"[Metrics] {args.agent} converged in {its} iterations")
            metrics_dir = "metrics"
            os.makedirs(metrics_dir, exist_ok=True)
            # grid_name = Path(grid).stem  # Extract just the filename without extension
            grid_name = Path(grid_fp).stem
            param_str = "_".join(f"{k}-{v}" for k, v in init_args.items())
            fname = f"{args.agent}_grid-{grid_name}_sigma-{env.sigma}_iter-{args.iter}_{param_str}_{uuid.uuid4().hex[:8]}.json"

            path = os.path.join(metrics_dir, fname)
        try:
            with open(path, "w") as mf:
                json.dump(agent.metrics, mf, indent=2)
                print(f"[Metrics] Saved convergence data to {path}")
        except Exception as e:
            print(f"[Metrics] ERROR saving metrics: {e}")

        text = buf.getvalue()
        metrics = {}
        for line in text.splitlines():
            m = re.match(r"\s*([a-z_]+)\s*:\s*([-+]?[0-9]*\.?[0-9]+)", line)
            if m:
                k, v = m.group(1), m.group(2)
                metrics[k] = int(v) if v.isdigit() else float(v)
        return metrics


## Loop to run experiments
def main():

    # Generate experiment csv

    CONFIG_PATH = "experiment_values.json"

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    defaults = config["defaults"]
    grids = config["grids"]
    experiments = config["experiments"]

    all_param_names = set()
    for agent_params in defaults.values():
        all_param_names.update(agent_params.keys())

    rows = []

    for agent, sweep in experiments.items():
        default_params = defaults[agent]

        for grid in grids:
            for param_name, values in sweep.items():
                for val in values:
                    params = deepcopy(default_params)
                    params[param_name] = val

                    row = {
                        "agent": agent,
                        "grid": grid,
                        "param_changed": param_name,
                        "param_value": val
                    }

                    for pname in sorted(all_param_names):
                        row[pname] = params.get(pname, float('nan'))

                    rows.append(row)

    df = pd.DataFrame(rows)


    df = df.loc[df["agent"] != "RandomAgent"]
    df.to_csv("experiment_results/experiment_table.csv", index = False)

    # df = pd.read_csv("experiment_results/experimental_table.csv")
    base_cfg = json.load(open("agent_config.json"))
    exp_defs = json.load(open("experiment_values.json"))

    rows = []
    for idx, row in df.iterrows():
        agent = row["agent"]
        grid  = row["grid"]
        print(f"{idx+1}: {agent} on {grid} | {row['param_changed']}={row['param_value']}")

        init_args, cli_args = {}, {}
        for c,v in row.items():
            if pd.isna(v) or c in {"agent","grid","param_changed","param_value"}:
                continue
            if c in {"episodes","iter"}:
                cli_args[c] = int(v)
            elif c == "sigma":
                cli_args[c] = float(v)
            else:
                init_args[c] = float(v)

        cfg = deepcopy(base_cfg)
        defaults_init = cfg[agent].get("init_args", {})
        cfg[agent]["init_args"] = {**defaults_init, **init_args}

        default_sigma = exp_defs["defaults"][agent].get("sigma", 0.0)
        sigma = cli_args.get("sigma", default_sigma)
        agent_start_pos = [1,13] if grid=="A1_grid" else [1,1]

        ns = Namespace(
            GRID=[f"grid_configs/{grid}.npy"],
            agent=agent,
            no_gui=True,
            sigma=sigma,
            fps=5,
            episodes=cli_args.get("episodes",
                                exp_defs["defaults"][agent].get("episodes",2000)),
            iter=cli_args.get("iter",
                            exp_defs["defaults"][agent].get("iter",2000)),
            random_seed=42,
            agent_start_pos=agent_start_pos
        )

        metrics = train_and_eval(ns, cfg)

        result = row.to_dict()
        result.update(metrics)
        result["sigma"] = sigma
        rows.append(result)

    out_df = pd.DataFrame(rows)
    out_df.to_csv("result_multi_experiment.csv", index=False)

if __name__ == "__main__":
    main()
import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict

METRICS_DIR = "metrics"
OUTPUT_DIR = "plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_filename(filename):
    """Extracts agent name, grid name, and parameter dictionary from filename"""
    base = filename[:-5]  # remove .json
    parts = base.split("_")
    agent_name = parts[0]

    params = {}
    grid = "unknown"

    for part in parts[1:]:
        if "-" in part:
            key, val = part.split("-", 1)
            if key == "grid":
                grid = val
            else:
                try:
                    val_eval = eval(val)
                except:
                    val_eval = val
                params[key] = val_eval
    return agent_name, grid, params

def load_metrics():
    data = defaultdict(lambda: defaultdict(list))  
    for fname in os.listdir(METRICS_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            path = os.path.join(METRICS_DIR, fname)
            with open(path, "r") as f:
                metrics = json.load(f)

            if not any(k in metrics for k in ("deltas", "mean_values", "rewards")):
                continue

            agent, grid, params = parse_filename(fname)
            data[agent][grid].append((params, metrics))
        except Exception as e:
            print(f"[Load Error] {fname}: {e}")
    return data

def find_varying_param(runs_by_grid):
    """Detects which single param varies across runs"""
    all_keys = set()
    all_values = defaultdict(set)

    for runs in runs_by_grid.values():
        for param_dict, _ in runs:
            for k, v in param_dict.items():
                all_keys.add(k)
                all_values[k].add(str(v))

    # Keep only keys with more than one distinct value
    varying = [k for k in all_keys if len(all_values[k]) > 1]
    return varying[0] if len(varying) == 1 else None

def plot_metric(agent, runs_by_grid, metric_key, ylabel, filename_suffix):
    param_key = find_varying_param(runs_by_grid)
    if not param_key:
        print(f"[Skip] {agent}: couldn't identify a single varying param for '{metric_key}'")
        return

    plt.figure(figsize=(10, 5))
    grid_styles = {
        "Maze": "-",       
    }

    for grid, runs in runs_by_grid.items():
        linestyle = grid_styles.get(grid, "-.")  
        for param_dict, metrics in runs:
            if metric_key not in metrics:
                continue
            label = f"{grid}, {param_key}={param_dict[param_key]}"
            plt.plot(metrics[metric_key], label=label, linestyle=linestyle)

    plt.title(f"{agent}: {ylabel} vs {param_key}")
    plt.xlabel("Episode" if metric_key == "rewards" else "Iteration")
    plt.ylabel(ylabel)
    plt.legend(fontsize="small", loc="best")
    plt.grid(True)
    plt.tight_layout()

    fname = f"{agent}_{param_key}_{filename_suffix}.png"
    out_path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(out_path)
    plt.close()
    print(f"[Plot] Saved {out_path}")


def plot_all(data):
    for agent, runs_by_grid in data.items():
        plot_metric(agent, runs_by_grid, "deltas", "Delta", "convergence")
        plot_metric(agent, runs_by_grid, "rewards", "Reward", "learning_curve")
        plot_metric(agent, runs_by_grid, "mean_values", "Mean V(s)", "value_curve")

if __name__ == "__main__":
    data = load_metrics()
    if not data:
        print("No valid metrics found.")
    else:
        plot_all(data)

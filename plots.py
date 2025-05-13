import os
import json
import matplotlib.pyplot as plt

METRICS_DIR = "metrics"
OUTPUT_DIR = "plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_metrics():
    all_data = {}
    for fname in os.listdir(METRICS_DIR):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(METRICS_DIR, fname)
        try:
            with open(path, "r") as f:
                metrics = json.load(f)

            if "deltas" in metrics or "mean_values" in metrics or "rewards" in metrics:
                base_name = fname[:-5]  # Strip '.json'
                agent_parts = base_name.split("_", 1)
                agent_name = agent_parts[0]
                param_str = agent_parts[1] if len(agent_parts) > 1 else "default"

                if agent_name not in all_data:
                    all_data[agent_name] = {}
                all_data[agent_name][param_str] = metrics

        except Exception as e:
            print(f"Failed to load {fname}: {e}")
    return all_data


def plot_convergence(all_data):
    for agent, variants in all_data.items():
        plt.figure(figsize=(10, 5))
        for param_str, metrics in variants.items():
            deltas = metrics.get("deltas", [])
            if not deltas:
                continue
            episodes = list(range(len(deltas)))
            plt.plot(episodes, deltas, label=param_str)

        plt.xlabel("Episode")
        plt.ylabel("Delta")
        plt.title(f"{agent}: Δ over Time")
        plt.grid(True)
        plt.legend(fontsize="small", loc="best")
        output_path = os.path.join(OUTPUT_DIR, f"{agent}_deltas.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"[Plot] Saved {output_path}")


def plot_learning_curves(all_data):
    for agent, variants in all_data.items():
        plt.figure(figsize=(10, 5))
        plotted = False
        ylabel = xlabel = title = ""

        for param_str, metrics in variants.items():
            if "mean_values" in metrics:
                plt.plot(metrics["mean_values"], label=param_str)
                ylabel = "Average Value"
                xlabel = "Iteration"
                title = f"{agent}: Mean V(s) over Iterations"
                plotted = True

            elif "rewards" in metrics:
                plt.plot(metrics["rewards"], label=param_str)
                ylabel = "Episode Reward"
                xlabel = "Episode"
                title = f"{agent}: Reward per Episode"
                plotted = True

        if not plotted:
            continue

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend(fontsize="small", loc="best")
        output_path = os.path.join(OUTPUT_DIR, f"{agent}_learning_curve.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"[Plot] Saved learning curve: {output_path}")



if __name__ == "__main__":
    data = load_metrics()
    if not data:
        print("No valid metrics found.")
    else:
        plot_convergence(data) 
        plot_learning_curves(data)       
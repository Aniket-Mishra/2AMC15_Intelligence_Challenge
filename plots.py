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
        with open(path, "r") as f:
            try:
                metrics = json.load(f)
                if "deltas" in metrics or "mean_values" in metrics or "rewards" in metrics:
                    agent_name = fname.split("_")[0]  
                    all_data[agent_name] = metrics
            except Exception as e:
                print(f"Failed to load {fname}: {e}")
    return all_data


def plot_per_agent(all_data):
    for agent, metrics in all_data.items():
        deltas = metrics.get("deltas", [])
        if not deltas:
            continue

        plt.figure(figsize=(8, 4))
        episodes = list(range(len(deltas)))
        plt.plot(episodes, deltas, label="Raw", color="blue", linewidth=1)
        plt.xlabel("Episode")
        plt.ylabel("Delta")
        plt.title(f"{agent}: Δ over Time")
        plt.grid(True)
        plt.legend()
        output_path = os.path.join(OUTPUT_DIR, f"{agent}_deltas.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"[Plot] Saved {output_path}")

def plot_learning_curves(all_data):
    for agent, metrics in all_data.items():
        plt.figure(figsize=(8, 4))

        if "mean_values" in metrics:
            # Value Iteration agent
            plt.plot(metrics["mean_values"], label="Avg V(s)", color="green")
            plt.ylabel("Average Value")
            plt.title(f"{agent}: Mean V(s) over Iterations")

        elif "rewards" in metrics:
            # QL agent
            plt.plot(metrics["rewards"], label="Reward", color="orange")
            plt.ylabel("Episode Reward")
            plt.title(f"{agent}: Reward per Episode")

        else:
            continue  # Skip agents with no relevant metric

        plt.xlabel("Iteration" if "mean_values" in metrics else "Episode")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f"{agent}_learning_curve.png")
        plt.savefig(output_path)
        plt.close()
        print(f"[Plot] Saved learning curve: {output_path}")


if __name__ == "__main__":
    data = load_metrics()
    if not data:
        print("No valid metrics found.")
    else:
        plot_per_agent(data) 
        plot_learning_curves(data)       
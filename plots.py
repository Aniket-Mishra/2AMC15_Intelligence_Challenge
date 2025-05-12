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
                deltas = metrics.get("deltas", [])
                if deltas:
                    agent_name = fname.split("_")[0]  
                    all_data[agent_name] = deltas
            except Exception as e:
                print(f"Failed to load {fname}: {e}")
    return all_data

def plot_per_agent(all_data):
    for agent, deltas in all_data.items():
        plt.figure(figsize=(8, 4))
        plt.plot(range(len(deltas)), deltas, marker='o', linewidth=1)
        plt.xlabel("Episode")
        plt.ylabel("Delta")
        plt.title(f"{agent}: Δ over Time")
        plt.grid(True)
        output_path = os.path.join(OUTPUT_DIR, f"{agent}_deltas.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"[Plot] Saved {output_path}")

if __name__ == "__main__":
    data = load_metrics()
    if not data:
        print("No valid metrics found.")
    else:
        plot_per_agent(data)        
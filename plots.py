import os
import json
import matplotlib.pyplot as plt

BASE_DIR = "metrics"
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_lines_from_folder(folder_path, metric_keys, folder_name, subfolder_name):
    for metric in metric_keys:
        plt.figure(figsize=(10, 5))
        files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
        found = False

        for fname in files:
            fpath = os.path.join(folder_path, fname)
            try:
                with open(fpath, "r") as f:
                    metrics = json.load(f)
                if metric in metrics:
                    # Use dashed line for maze
                    linestyle = '--' if 'grid-Maze' in fname else '-'
                    label = fname.replace(".json", "")
                    plt.plot(metrics[metric], label=label, linestyle=linestyle)
                    found = True
            except Exception as e:
                print(f"[Error] {fname}: {e}")

        if found:
            plt.title(f"{folder_name}/{subfolder_name}: {metric} vs Iteration")
            plt.xlabel("Iteration")
            plt.ylabel(metric)
            plt.legend(fontsize="small", loc="best")
            plt.grid(True)
            plt.tight_layout()
            out_name = f"{folder_name}_{subfolder_name}_{metric}.png"
            plt.savefig(os.path.join(OUTPUT_DIR, out_name))
            plt.close()
            print(f"[Plot] Saved {out_name}")
        else:
            plt.close()



def process_all_folders():
    top_folders = ["MC", "Q", "Val"]

    for folder_name in top_folders:
        full_folder = os.path.join(BASE_DIR, folder_name)
        if not os.path.isdir(full_folder):
            continue

        subfolders = [d for d in os.listdir(full_folder) if os.path.isdir(os.path.join(full_folder, d))]
        for subfolder in subfolders:
            subfolder_path = os.path.join(full_folder, subfolder)
            if folder_name in ["MC", "Q"]:
                metric_keys = ["deltas", "rewards"]
            else:  # Val
                metric_keys = ["deltas", "mean_values"]
            plot_lines_from_folder(subfolder_path, metric_keys, folder_name, subfolder)

if __name__ == "__main__":
    process_all_folders()

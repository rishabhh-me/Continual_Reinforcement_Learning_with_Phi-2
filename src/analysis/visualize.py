import argparse
import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath("."))

from src.analysis.utils import load_json, ensure_dir

def plot_subgoal_taxonomy(metrics, output_dir):
    taxonomy = metrics.get("taxonomy", {})
    full_counts = taxonomy.get("full", {})

    if not full_counts: return

    # Top 20 subgoals
    df = pd.DataFrame(list(full_counts.items()), columns=["Subgoal", "Count"])
    df = df.sort_values("Count", ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=df, y="Subgoal", x="Count", palette="viridis")
    plt.title("Top 20 Subgoals")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "subgoal_counts.png"))
    plt.close()

def plot_failures(metrics, output_dir):
    taxonomy = metrics.get("taxonomy", {})
    if not taxonomy: return

    df = pd.DataFrame(list(taxonomy.items()), columns=["Type", "Count"])

    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x="Type", y="Count", palette="rocket")
    plt.title("Failure Modes")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "failure_modes.png"))
    plt.close()

def plot_gating(metrics, output_dir):
    hist = metrics.get("query_position_hist", [])
    if not hist: return

    # In axis_c_gating.py, we only saved the counts: np.histogram(...)[0].tolist()
    counts = hist

    plt.figure(figsize=(8, 6))
    plt.bar(range(len(counts)), counts)
    plt.xlabel("Episode Progress (0=Start, 1=End)")
    plt.ylabel("Query Count")
    plt.title("Planner Query Distribution")
    plt.savefig(os.path.join(output_dir, "gating_distribution.png"))
    plt.close()

def plot_representation(data, output_dir):
    coords = data.get("coords", [])
    subgoals = data.get("subgoals", [])

    if not coords: return

    coords = np.array(coords)

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], alpha=0.7)

    # Annotate random subset
    indices = np.random.choice(len(subgoals), min(20, len(subgoals)), replace=False)
    for i in indices:
        plt.text(coords[i, 0], coords[i, 1], subgoals[i], fontsize=8)

    plt.title("Subgoal Representation (PCA)")
    plt.savefig(os.path.join(output_dir, "subgoal_embeddings.png"))
    plt.close()

def visualize(metrics_dir, output_dir):
    print(f"Generating visualizations from {metrics_dir} to {output_dir}...")
    ensure_dir(output_dir)

    # Load Metrics
    axis_a = os.path.join(metrics_dir, "axis_a_metrics.json")
    axis_b = os.path.join(metrics_dir, "axis_b_metrics.json")
    axis_c = os.path.join(metrics_dir, "axis_c_metrics.json")
    axis_d = os.path.join(metrics_dir, "axis_d_embeddings.json")

    if os.path.exists(axis_a):
        plot_subgoal_taxonomy(load_json(axis_a), output_dir)

    if os.path.exists(axis_b):
        plot_failures(load_json(axis_b), output_dir)

    if os.path.exists(axis_c):
        plot_gating(load_json(axis_c), output_dir)

    if os.path.exists(axis_d):
        plot_representation(load_json(axis_d), output_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", type=str, default="analysis_output/metrics")
    parser.add_argument("--output_dir", type=str, default="analysis_output/figures")

    args = parser.parse_args()
    visualize(args.metrics_dir, args.output_dir)

if __name__ == "__main__":
    main()

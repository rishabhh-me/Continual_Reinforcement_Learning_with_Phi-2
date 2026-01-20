import argparse
import os
import sys
import json
import glob
import numpy as np
from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE # TSNE is slow, let's use PCA for now or TSNE if small

sys.path.append(os.path.abspath("."))

from src.analysis.utils import load_json, save_metrics, ensure_dir

def analyze_representation(trace_files, output_file):
    print(f"Analyzing representation from {len(trace_files)} files...")

    unique_subgoals = set()

    for filepath in trace_files:
        with open(filepath, "r") as f:
            for line in f:
                trace = json.loads(line)
                for step in trace["steps"]:
                    if step.get("replan", False) or step["step"] == 0:
                        # Use the text description
                        # Note: In collect_traces.py, we saved "subgoal_text".
                        # If "subgoal_text" is missing or None, skip.
                        st = step.get("subgoal_text", "")
                        if st and st != "None":
                            unique_subgoals.add(st)

    subgoals_list = list(unique_subgoals)
    print(f"Found {len(subgoals_list)} unique subgoals.")

    if not subgoals_list:
        print("No subgoals found.")
        return

    # Embed
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading SentenceTransformer...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(subgoals_list)
    except ImportError:
        print("SentenceTransformer not found. Using Random Embeddings.")
        embeddings = np.random.randn(len(subgoals_list), 384)
    except Exception as e:
        print(f"Embedding failed: {e}. Using Random Embeddings.")
        embeddings = np.random.randn(len(subgoals_list), 384)

    # Dimensionality Reduction
    coords = []
    if len(subgoals_list) > 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings).tolist()
    else:
        coords = [[0,0]] * len(subgoals_list)

    # Save
    data = {
        "subgoals": subgoals_list,
        "coords": coords,
        # "embeddings": embeddings.tolist() # Too large maybe?
    }

    save_metrics(data, output_file)
    print(f"Saved Axis D data to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="analysis_output/metrics/axis_d_embeddings.json")

    args = parser.parse_args()

    trace_files = glob.glob(os.path.join(args.trace_dir, "**/*.jsonl"), recursive=True)
    analyze_representation(trace_files, args.output)

if __name__ == "__main__":
    main()

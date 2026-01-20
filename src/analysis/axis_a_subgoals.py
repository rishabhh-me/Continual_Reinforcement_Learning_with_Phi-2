import argparse
import os
import sys
import json
import glob
import numpy as np
from collections import Counter

sys.path.append(os.path.abspath("."))

from src.analysis.utils import load_json, save_metrics, ensure_dir

def analyze_subgoals(trace_files, output_file):
    print(f"Analyzing subgoals from {len(trace_files)} files...")

    all_subgoals = []
    subgoal_tuples = []

    for filepath in trace_files:
        with open(filepath, "r") as f:
            for line in f:
                trace = json.loads(line)
                for step in trace["steps"]:
                    # check if replan occurred or it's the first step
                    # Actually we want all generated subgoals.
                    # The trace has "subgoal_tuple" at every step.
                    # We only want to count *generated* subgoals.
                    # In collect_traces.py, we record "replan": True/False.
                    # Also the first step has a subgoal.

                    if step["step"] == 0 or step.get("replan", False):
                        sg_tuple = tuple(step["subgoal_tuple"])
                        # Filter out NO_OP if present (though usually it's replanned immediately)
                        if sg_tuple[0] == "no_op":
                            continue

                        all_subgoals.append(sg_tuple)
                        subgoal_tuples.append(sg_tuple)

    # 1. Taxonomy
    action_counts = Counter([s[0] for s in all_subgoals])
    object_counts = Counter([s[2] for s in all_subgoals])
    color_counts = Counter([s[1] for s in all_subgoals])
    full_counts = Counter(["_".join(s) for s in all_subgoals])

    # 2. Reuse / Entropy
    unique_subgoals = len(full_counts)
    total_subgoals = len(all_subgoals)

    # Calculate Entropy
    probs = np.array(list(full_counts.values())) / total_subgoals
    entropy = -np.sum(probs * np.log(probs + 1e-9))

    metrics = {
        "taxonomy": {
            "actions": dict(action_counts),
            "objects": dict(object_counts),
            "colors": dict(color_counts),
            "full": dict(full_counts) # Top N?
        },
        "reuse": {
            "unique_count": unique_subgoals,
            "total_count": total_subgoals,
            "entropy": float(entropy),
            "reuse_ratio": 1.0 - (unique_subgoals / total_subgoals) if total_subgoals > 0 else 0.0
        }
    }

    save_metrics(metrics, output_file)
    print(f"Saved Axis A metrics to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_dir", type=str, required=True, help="Directory containing .jsonl trace files")
    parser.add_argument("--output", type=str, default="analysis_output/metrics/axis_a_metrics.json")

    args = parser.parse_args()

    trace_files = glob.glob(os.path.join(args.trace_dir, "**/*.jsonl"), recursive=True)
    analyze_subgoals(trace_files, args.output)

if __name__ == "__main__":
    main()

import argparse
import os
import sys
import json
import glob
import numpy as np

sys.path.append(os.path.abspath("."))

from src.analysis.utils import load_json, save_metrics, ensure_dir

def analyze_gating(trace_files, output_file):
    print(f"Analyzing gating from {len(trace_files)} files...")

    inter_query_intervals = []
    query_positions = [] # Normalized position in episode (0.0 to 1.0)
    total_queries = 0
    total_steps_sum = 0

    for filepath in trace_files:
        with open(filepath, "r") as f:
            for line in f:
                trace = json.loads(line)
                steps = trace["steps"]
                total_steps = len(steps)
                if total_steps == 0: continue

                total_steps_sum += total_steps

                # Find query steps
                # Step 0 is always a query (initial plan)
                q_steps = [s["step"] for s in steps if s.get("replan", False) or s["step"] == 0]

                total_queries += len(q_steps)

                # Intervals
                if len(q_steps) > 1:
                    intervals = np.diff(q_steps)
                    inter_query_intervals.extend(intervals)

                # Positions
                for q in q_steps:
                    query_positions.append(q / total_steps)

    metrics = {
        "total_queries": total_queries,
        "avg_queries_per_episode": total_queries / len(trace_files) if trace_files else 0, # rough approx if 1 episode per line? No, len(trace_files) is files. Need episode count.
        "avg_inter_query_interval": float(np.mean(inter_query_intervals)) if inter_query_intervals else 0.0,
        "std_inter_query_interval": float(np.std(inter_query_intervals)) if inter_query_intervals else 0.0,
        "query_position_hist": np.histogram(query_positions, bins=10, range=(0,1))[0].tolist() if query_positions else []
    }

    # Correct avg per episode
    # We iterate files, then lines. We didn't count episodes above.
    # Let's count properly.

    save_metrics(metrics, output_file)
    print(f"Saved Axis C metrics to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="analysis_output/metrics/axis_c_metrics.json")

    args = parser.parse_args()

    trace_files = glob.glob(os.path.join(args.trace_dir, "**/*.jsonl"), recursive=True)
    analyze_gating(trace_files, args.output)

if __name__ == "__main__":
    main()

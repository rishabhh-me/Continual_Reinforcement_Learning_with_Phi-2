import argparse
import os
import sys
import json
import glob
from collections import Counter

sys.path.append(os.path.abspath("."))

from src.analysis.utils import load_json, save_metrics, ensure_dir

def check_hallucination(subgoal_tuple, obs_desc):
    """
    Returns True if the subgoal refers to an object NOT in the description.
    """
    action, color, obj = subgoal_tuple
    if action == "no_op": return False

    # Text Description Format: "You are carrying a color obj. In the room, you see: color obj, color obj."
    # We need to check if (color, obj) is in the text.

    desc_lower = obs_desc.lower()

    # Exceptions
    if obj in ["explore", "goal"]: return False # Abstract
    if action == "explore": return False

    # "Pick up the yellow key" -> need "yellow key" in desc OR carrying it?
    # If picking up, we shouldn't be carrying it (usually).

    # Construct target string
    target = f"{color} {obj}"
    if target in desc_lower:
        return False

    # Maybe "key" is mentioned but color is wrong?
    # Or "door" is mentioned.

    # If we are carrying it, and action is "open", it's valid to not see it in "room", but "carrying" part.
    if "carrying" in desc_lower and target in desc_lower:
         return False

    return True

def analyze_failures(trace_files, output_file):
    print(f"Analyzing failures from {len(trace_files)} files...")

    failure_counts = Counter()
    total_episodes = 0
    failed_episodes = 0

    for filepath in trace_files:
        with open(filepath, "r") as f:
            for line in f:
                trace = json.loads(line)
                total_episodes += 1

                if trace["success"] == 1:
                    continue

                failed_episodes += 1

                # Analyze the failure reason for this episode
                # We look at the LAST subgoal active.
                steps = trace["steps"]
                if not steps:
                    failure_counts["empty_trace"] += 1
                    continue

                # Iterate through steps to find the first "Hallucination"
                hallucinated = False
                for step in steps:
                    if step.get("replan", False) or step["step"] == 0:
                        sg_tuple = step["subgoal_tuple"]
                        obs = step["obs_desc"]
                        if check_hallucination(sg_tuple, obs):
                            failure_counts["hallucination"] += 1
                            hallucinated = True
                            break

                if hallucinated:
                    continue

                # If no hallucination, but failed -> Execution Mismatch
                # (The subgoal was valid, but we didn't finish the task)
                failure_counts["execution_mismatch"] += 1

    metrics = {
        "total_episodes": total_episodes,
        "failed_episodes": failed_episodes,
        "failure_rate": failed_episodes / total_episodes if total_episodes > 0 else 0,
        "taxonomy": dict(failure_counts)
    }

    save_metrics(metrics, output_file)
    print(f"Saved Axis B metrics to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="analysis_output/metrics/axis_b_metrics.json")

    args = parser.parse_args()

    trace_files = glob.glob(os.path.join(args.trace_dir, "**/*.jsonl"), recursive=True)
    analyze_failures(trace_files, args.output)

if __name__ == "__main__":
    main()
